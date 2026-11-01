---
title: "PipeLink：契约驱动的 GPU kernel 组件链接"
published: 2026-11-02T00:20:00+08:00
description: "PipeLink 把融合 kernel 的各个组件编译成带契约的 LTO 对象，链接时检查接口、用 SMT 求解一份 SM 资源放置、生成组件间的 tensor 搬运，再由 nvJitLink 把放置折成常量。51 个配置与单编译单元相差 0.2% 以内，混合 CuTe attention 的 decode megakernel 快 2.16×。"
tags: [PipeLink, CUDA, LTO, 编译链接, Megakernel, MoE, Attention]
category: Kernel
draft: false
---

一个高性能 attention kernel 不是一个算子，而是一组同时跑在同一个 SM 上的组件：load warp 搬 tile，两个 consumer warpgroup 算 MMA 和 online softmax，epilogue warp 写回，context-parallel 版本还多两个 warp 往别的 GPU 推 KV。它们共享同一块 shared memory、同一组 named barrier、同一个寄存器文件，靠 mbarrier ring 互相递数据。megakernel 更进一步，把整个 decode step 或训练 step 塞进一个 persistent kernel。

问题在于，这些组件没法单独编译、发布、替换。它们把“自己在这块共享空间里的位置”硬编码进了源码：`SharedStorage` 结构体把每个 buffer 和 barrier 放在固定偏移，named barrier 是枚举常量，角色靠 warp index 和常量比较来选，`setmaxnreg` 传的是写死的预算。想用别的库里的算子，只有两条路——fork 源码改位置，或者拆成两个 kernel 用 global memory 衔接。

PipeLink 想把“目标文件 + 链接器”这一套搬到 kernel 组件上：每个组件单独编译成 LTO 对象，同时携带一份**契约**说明它需要什么、承诺什么；链接器检查契约之间是否自洽，为所有组件**联合求解**一份 SM 资源放置，生成组件之间的数据搬运，最后交给厂商的 LTO 后端把放置折成常量。论文是 [PipeLink: Contract-Guided Linking of GPU Kernel Components](https://arxiv.org/abs/2610.37200)（Jize Guo、Xupeng Miao，2026，cs.DC），代码在 [PKU-DAIR/PipeLink](https://github.com/PKU-DAIR/PipeLink)，Apache-2.0。

> **五分钟版**
>
> - **要解决什么**：融合 kernel 的组件是“绝对寻址”的——每个组件都从 shared memory 偏移 0、barrier 1、warp 0 开始写，合到一起必然撞车，所以只能 fork 源码或拆 kernel。
> - **契约**：组件按 handle 声明资源（要多少 shared memory、几个 mbarrier 槽、几个 warp），声明生命周期、协议和“事实”（tile size 这类别人要依赖的常量），不声明地址。
> - **链接**：44 条检查规则先配对端点；再用 Z3 解一个约束问题，一次性定下所有 handle 的基址、ring 深度和寄存器预算；然后生成 stub、跳板和搬运代码，由 nvJitLink 完成 device link。
> - **为什么零开销**：放置通过 handle 函数读，链接器把它定义成返回常量，LTO 内联后折成立即数，`smem_base()` 变成 `1024`。51 个配置与单编译单元相差 0.2% 以内，很多 kernel 的机器码完全相同。
> - **能换来什么**：CuTe attention 塞进 ThunderKittens decode megakernel，4096 token 上下文下快 2.16×；四个团队各管一块的 MoE 训练 megakernel 与原单体持平；一个冻结的 attention core 变出七个变体，稀疏版比 FlexAttention 快最多 38×；Qwen3-32B 的 GEMM + SwiGLU 在 B300 上比 cuBLAS + 编译好的 SwiGLU 快 1.20×。
> - **边界**：只检查声明不检查实现，manifest 说谎照样过关；生成的代码只在 CTA 内搬数据、不做算术；不加速组件本身；一个包绑定一个架构和一个 CUDA 工具链。

## 组件为什么“不可重定位”

论文拿 Hopper 上的 context-parallel attention 当贯穿例子。它从 FlashAttention-3 派生：一个 load warp 把 Q/K/V tile 搬进 shared memory，两个 consumer warpgroup 用 online softmax 算 attention，一个 epilogue warp 写输出，再加两个 warp 把本地 KV shard 推给其他 GPU。十二个 warp 同时跑，通过 mbarrier ring、named barrier 和跨 GPU 的 readiness counter 互相递活。

这些交接全都有物理落点，而落点在源码里是常量。论文的概括很准确：今天的组件是**绝对寻址（absolute-addressed）**的，它们在一小块被所有组件共享的片上地址空间里写死了自己的位置。FlashAttention、CUTLASS 声明一个 `SharedStorage` 结构体铺开所有 buffer 和 barrier；named barrier 用枚举编号；角色靠 warp index 和常量比较来选；`setmaxnreg` 传常量预算。ThunderKittens 的 megakernel 也把固定的 page 和 semaphore 布局编进每个 operator。

一个团队拥有整个 kernel 时，这没什么问题，反而让编译器好优化。麻烦出现在复用的时候：独立写出来的两个组件都从偏移 0、barrier 1、warp 0 起步，原样拼在一起按构造就会冲突，必须有人拿着全部组件的知识重新推导每个位置。

### 两条老路：fork 或 split

| 路线 | 做法 | 代价 |
| :--- | :--- | :--- |
| Fork | 把各组件的源码并进一个 translation unit，手工重推所有共享赋值 | 每个 fork 都是一个新的、要单独维护的 kernel；跨库算子等于按目标 megakernel 的 page / semaphore / 指令约定重写一遍 |
| Split | 每个组件留在自己的 kernel 里不改，用 global memory 衔接 | 每个边界都要付 launch、显存往返和失去的 overlap |
| 模板库 | 只在作者预留的扩展点上组合（mask、score 函数、变体模板） | 覆盖不到作者没预料到的组合 |

Megakernel 走的是 fork：一个团队、一个库、一个编译单元，把 CuTe 的 attention 放进 ThunderKittens 的 megakernel 就得手工移植；新的 attention 变体往往从一个专家 kernel fork 出来；连“用 coding agent 给每个模型写一个新 megakernel”这类工作，本质上也是自动化 fork。

生产环境的 context-parallel attention 走的是 split：每个 KV block 调一次未改动的 FlashAttention kernel，块之间的跨 GPU 搬运交给其他 stream 上的集合通信，部分输出和 log-sum-exp 的修正在 kernel 外面单独做。复用保住了，但每个 kernel 边界都在付 launch、访存和失去的 overlap。

PipeLink 的判断是：这两条路都不是必须的，缺的是组件之间的接口和能对接口动手的链接器。

## 符号链接为什么不够

CUDA 其实早就有分离编译：relocatable device code 加 LTO 能把独立编译的 device function 合成一个 kernel，性能与单编译单元相当，NVIDIA 自己的库（cuFFT、cuFFTDx）就这么发。那为什么不能直接用？

因为 device link 只认符号：它记录一个片段定义了哪些名字、需要哪些名字，仅此而已。它不知道这个片段占了多少 warp、多少 shared memory、哪些 barrier、多少寄存器，也不知道它的邻居编译时依赖了哪些事实。链接器会愉快地接受一个语义上已经不成立的组合。

论文给了一个很能说明问题的实验。他们把一个 MoE 训练 megakernel 拆给四个 owner：framework 管任务解释器、cluster launch control 和 launch 几何，另外三个分别管通信（NVLink 上的 dispatch / combine）、grouped GEMM 和 SwiGLU。四个 owner 之间会按对方的 tile 几何来调度，还有一个三 owner 共享的计数器。

然后他们一次只改一个 owner 的一个几何常量，只重编那一个 owner，再和另外三个没动过的对象做 device link。五种组合，符号链接**全部接受**：三个 kernel 挂起，两个跑完不报错但有一部分输出没写。

论文 Figure 2 的例子是通信 owner 把 `DISPATCH_Nb` 从 512 改成 256。按符号链接，framework 和 GEMM 还按 512 排，kernel 挂死；按契约链接，新值作为“事实”传播到所有消费者，通过。这就是动机：**静默失败比链接错误更贵**，而且这种失败很难在 code review 里看出来。

## 契约里有什么

PipeLink 把组件分成三类：

- **role**：一组 warp 加一个入口函数，比如 TMA loader、MMA mainloop、通信 stage。
- **callable**：跑在调用者线程上的 device function，比如 attention mask、score 修正、megakernel 解释器里的算子体。
- **tensor producer / consumer**：导出或导入一块片上 tile 的 role 或 callable。

一个源文件可以定义多个组件，组件可以用普通 CUDA、CUTLASS/CuTe、ThunderKittens 或 TileLang 写。作者把组件打成一个 **package**：厂商编译器产出的 LTO 对象，加上记录契约的 manifest。打包时会编译两次——一次求值那些构成契约的注解，一次产出对象。之后链接器把这个对象当成不透明的二进制，不读也不改它的指令，所以组件可以在没有源码的情况下分发。

契约的词汇表只有两列：

| 声明（Requires） | 承诺（Promises） |
| :--- | :--- |
| 每个 role 的 warp 数 | role 及其入口函数 |
| shared memory、barrier、tensor memory、counter，各自带生命周期 | channel endpoint 及其协议 |
| 它调用的 callable | 它导出的 callable |
| 它 import 的 tensor（demand） | 它 export 的 tensor（placement） |
| 它调度时依赖的事实 | 它拥有的事实，比如 tile size |
| caller buffer 与通信域 | 它构造的 host 参数 |

这套词汇有四个设计点，值得单独说：

- **按 handle 认领，不按地址**。一条 claim 只说“我需要 32 KiB shared memory / 4 个 mbarrier 槽 / 8 个 warp”，从不说在哪。
- **生命周期让不相交的用途共享存储**。claim 可以活整个 kernel、活 megakernel 指令流的某个 phase，或活在一个 tensor 连接的两个事件之间——这样才能让 epilogue 的 scratch 复用 mainloop 的 operand ring。
- **同步按协议描述，不按代码描述**。一个 mbarrier ring 声明它的 arrival count 是参与 role 宽度的函数、transaction 字节数、以及哪个 role 负责初始化，于是两个端点能在不看彼此函数体的情况下被检查是否一致。想从代码里反推这些性质，既要分析不透明对象，也仍然会丢掉意图（这块 buffer 装的是哪个 tile、什么时候能复用）。
- **事实是有类型的常量**。tile size 这类东西由一个组件拥有，其他组件按它调度。消费者不把事实编进自己的代码，而是声明一个“来自事实的 kernel 参数”；链接器检查所有声明一致，再把 owner 的值写进 kernel 参数。于是改 tile size 不需要重编任何消费者——这正好是符号链接做不到的那件事。

## 组件怎么变得可重定位

可重定位的关键是一个很小的技巧：组件通过 **handle 函数**读取每一个链接期决定。handle 函数是按 handle 命名的外部 device function，在组件里只有声明、没有定义。下段节选自论文 Figure 3(a) 的 CUDA 消费者，省略了部分参数：

```cpp
PL_STAGE("swiglu.cuda", "1.0.0");
PL_ROLE(epilogue, epilogue_role, THREADS / 32);
PL_CLAIM_SMEM("output_smem", OutputTile, ...);
PL_IMPORT_TENSOR(acc, import_acc, /* placement / demand / 借用资源 见论文 Figure 3(a) */ ...);

// 组件代码通过 handle 拿位置，而不是写死偏移
auto& tile = *(OutputTile*)(smem + output_smem());
import_acc(source, chunk, acc, scratch);
```

`output_smem()` 在对象里是一个未解析的调用。shared memory 基址、mbarrier 槽、named barrier 编号、warp 偏移、ring 深度、counter 槽，全部这么读。所以对象里留下的是“未解析的调用”而不是常量，很像位置无关代码通过表读地址。区别是这张表不要钱：链接器把 handle 函数定义成返回一个常量，LTO 会把它折掉（下一节展开）。

论文比较了三种替代方案，每种都丢掉一个必要条件：

| 方案 | 对象能否保持不变 | 是否零运行时开销 | 结论 |
| :--- | :---: | :---: | :--- |
| 模板 / 宏参数 | ✗ 每个组合都要重编，也就要源码 | ✓ | 不满足“二进制分发” |
| kernel 参数 / constant memory 传放置 | ✓ | ✗ 每个偏移变成运行时 load，地址运算无法常量折叠 | 不满足零开销 |
| 二进制里打 relocation 补丁（CPU 链接器的做法） | ✓ | ✓ | named barrier 编号、warp 范围、寄存器预算不是地址，是指令操作数或控制流，没有对应的重定位项 |

handle 函数同时满足三条：对象不变、绑定按组合选、运行时零成本。

## 打包：作者要写多少

注解就写在组件自己的源文件里，紧挨着被描述的代码，由编译器在打包时求值。作者只写编译器看不见的东西：语义承诺，比如这个 export 装的是哪个 tile、这个 ring 遵循什么协议、哪个 role 初始化 barrier、一块 buffer 活多久。编译器能观测到的东西则由打包过程直接采集：参数和描述符类型的大小、对齐、字段偏移，对象定义和引用了哪些 device 符号，编译产物里的 launch attribute，以及把每个枚举型 layout map 拟合成闭式表达式——拟合不精确就直接拒绝这个 package。

论文给了两组“作者成本”的数字，挺能说明声明式接口和手写的差距：

- 同一个 SwiGLU 消费者，CUDA 版写 10 条 device 侧注解、共 38 行；TileLang 版**一条注解都没写**，只有 14 行前端调用（指明要 import 哪个 export、chunk 大小、是否特化、完成策略），其余从 producer 的契约或编译产物里取。两个消费者共用同样的 9 条 host 注解和 host provider，消费者专属源码分别是 163 行和 56 行。
- 其他案例里，一个新的 attention load 或通信组件要 13–20 条注解语句，一个 megakernel 算子 11–15 条；MoE megakernel 的四个 owner 一共写了 327 条语句，对应 3,482 行组件与 host provider 源码。

TileLang 前端值得一提：tensor import 的 demand 是从目标 fragment 的 layout 里推出来的，所以作者一个放置事实都不用写。它还可以选择性地“特化”到某个 producer——采用对方 manifest 里公布的寄存器 layout。特化之后仍然能和任何其他 producer 链接，只是链接器这时可以选直接寄存器拷贝而不是经 shared memory 中转。

## 链接的四步

一条 pipeline manifest 列出这个组合用到的所有 package。链接分四步：

```text
provider.plink   caller.plink   ...
       │              │
       └────── pipeline.json ──────┐
                                   ▼
                     1. 检查契约（44 条规则）
                                   ▼
                     2. 求解放置（Z3 / SMT）
                                   ▼
                     3. 生成 stub、跳板与搬运代码
                                   ▼
                     4. nvJitLink 做 device link
                                   ▼
                        一个 cubin + 每一步的记录
```

### 1. 检查

链接器按名字配对端点，检查每对端点在 memory space、scope、tile、layout、protocol 上是否一致；对 mbarrier ring 还要检查 arrival count、transaction 字节数和哪个 warp 负责初始化。callable 要签名匹配；事实要在 owner 和消费者之间一致；cluster shape 这类拓扑要对所有组件取交集。

拒绝会给出 44 条规则中的一条，加上冲突的事实：system scope 的 counter producer 配了一个 grid scope 的 consumer，报 `L-PAIR`；一个声明了 544 列的 tensor memory claim 放不下，报 `L-OVERLAP`。

没人调用的 export 会被丢掉，就像普通链接器丢掉未引用的 section。所以一个 package 可以服务多个组合，每个组合只调它需要的那部分 export——MoE 那 6 个 package 能链进 24 个 kernel，靠的就是这一点。

### 2. 求解放置

经典链接器靠拼接 section 来重定位，这在 SM 上不成立：资源空间又小又耦合。一个 thread block 最多 227 KiB shared memory、32 个 warp、16 个 named barrier；warp 被哪个 role 占决定了它属于哪个寄存器组，寄存器组之间共享一个寄存器文件，buffer 只有在生命周期不相交时才能共享存储。

所以 PipeLink 为每个组合解一个约束问题（Z3）：

- **变量**：每个域里每个 claim 的基址、留给链接器决定的 ring 深度、每个寄存器组的预算。
- **约束**：容量、对齐、role 对 warp 的精确覆盖、不重叠。两条 claim 只有在 phase 互斥、或者它们在一个 tensor 连接上的生命周期有序时才能重叠——所以一旦 accumulator 就绪，epilogue 的 scratch 就能复用 mainloop 的 operand ring。
- **目标**：先偏好更深的 ring 和更大的寄存器预算，再偏好更小的占用。求解器是确定性的，同一个组合永远得到同一个放置。

为什么不用贪心？因为生命周期把放置变成了装箱问题，而装箱的结果取决于放置顺序。MoE megakernel 里，按词法顺序或按大小递减做 first-fit 都找不到最小的 shared memory 占用；求解器找到了一个，最后还剩 232,448 字节里的 1,440 字节。这些组合的 handle 数是 4–42 个，论文说求解器对每个组合都能在 1.3% 的间隙内证明目标最优。

两条边界也很重要：

- **PipeLink 从不重排组件的调度**。warp specialization 编译器会为整个 kernel 选流水线和角色分配，但那需要源码；这里的 owner 已经针对自己的硬件调好了。保持每个调度不变，链接后 kernel 的性能就能从各部分的性能推出来，链接器也不必进入组件的正确性论证。
- **寄存器由厂商后端分配。** device link 之后 PipeLink 读一遍链接结果的寄存器用量，如果寄存器组放不下就重解一次；还放不下就以 `L-REGS` 失败。

### 3. 为什么零开销

链接器生成三段代码：把每个 handle 函数定义成返回求解常量的 **stub**、把每个调用点连到对应 export 的**跳板**、以及一个分配 warp 到 role、并给每个寄存器组设一次预算的 **dispatch kernel**。三段都编译成 LTO IR，和组件一起通过 nvJitLink 做 device link。

后端会内联 stub，把常量折进组件的地址运算，于是 `smem_base()` 就变成了立即数 `1024`。得到的是“把组件写进一个源文件、用同样放置”时程序员会拿到的同一份代码。

用函数指针分发更省事，但间接调用不会被内联，常量折不掉，而且所有 role 都得按最挑剔的那个角色的寄存器预算来跑——这正是零开销设计要避免的。

### 4. 生成的 tensor 搬运

GEMM mainloop 把 accumulator 留在 tensor memory（Blackwell）或寄存器（Hopper、消费级 Blackwell）里，而 epilogue 想让自己的每个线程持有特定元素。手写这层搬运需要知道 producer 的 layout，而这正是契约想去掉的耦合。

做法是让两端各说各的：producer 的 export 公布 tile 的 **placement**（每个元素住在哪，TMEM 的 lane 和列、shared memory 地址、或者线程号加寄存器号）；consumer 的 import 声明 **demand**（它的每个线程想要哪个元素）。两者都是打包时拟合出的精确映射。

链接器然后从便宜到通用试几条路线：从 tensor memory 直接读进目标寄存器 → placement 和 demand 恰好一致时直接寄存器拷贝 → warp 内 shuffle → 借 consumer 的 scratch shared memory 中转。每条候选都要过一遍硬件原语的模型，检查是否每个元素都被精确送达、无竞争、顺序正确，第一条通过的用 TileLang 降到 CUDA。搬运只搬值，从不做转换；它的 scratch 也只是一条普通的 claim。

论文也解释了为什么不走另两条路：永远经 shared memory 中转对任何一对都成立，但要多付一次往返（5.5 节量化了）；约定一种固定的交换 layout 则要求所有 producer/consumer 事先一致，等于把耦合请回来。至于通用程序综合，能找到更聪明的路线，但结果难界定也难检查。

## 通信与宿主接口

通信算法留在组件代码里。一个跨 GPU 搬数据的组件声明它加入的通信域、它需要 caller 借出的对称分配上的哪些视图（peer 地址、multicast 地址）、以及跨 GPU 就绪用的 system-scope 计数器。链接器检查同一块分配的所有 owner 对域和视图是否一致，像对待普通 claim 一样绑定 counter 槽，再把 caller 的地址传给各 owner。

它**不合成集合通信**：ring push、带路由的 dispatch、reduce-scatter，协议都和算法强相关，设计它们的 owner 才最清楚怎么写。链接器的职责是让这样的组件在共享 kernel 里成为一个规矩的邻居。

宿主侧的准备（比如构造 TMA descriptor）以一个小型编译产物交付，藏在 C 接口后面。session 打开时，每个 owner 的 provider 从 caller 借出的 buffer 里填自己那段 kernel 参数；其他 owner 需要的事实走 manifest，从不走共享头文件。打开后的 session 启动时不需要 host 计算，因此可以被 CUDA graph 捕获。

## 评测：四个案例

下面的数字全部来自论文和仓库文档，我没有在本地复现，口径也照论文的表。

实现上，PipeLink 编译器约 24,000 行 Python：用 nvcc 打包，Z3 解放置，TileLang 降级生成的搬运，nvJitLink 做链接（CUDA 13.3），另有一个约 5,400 行的 TileLang 前端。

硬件是 H200（Hopper，`sm90`）、B300（Blackwell，`sm103`）和 RTX 5090（消费级 Blackwell，`sm120`），每台机器锁在它能同时跑满 dense BF16 和 FP8 矩阵乘的最高 SM 频率：1230 MHz、1155 MHz、1770 MHz。方法学上，每个链接出来的 kernel 都有一个单编译单元对照：同样的组件源码、同样的生成代码、同样的放置，放进一个 translation unit 编译；两者在同一进程、同一输入上交替跑，中间还重载一次 module。所有比值都是“链接版 / 对照”的吞吐比。

正式结论是：**51 个计时配置里，链接版都在单编译单元版的 0.2% 以内**；七个 attention 组合、混合库 megakernel、以及 H200 和 RTX 5090 上的 GEMM + SwiGLU，链接版和对照版的**归一化机器码完全相同**。所以这四个案例量的是“组合能带来什么”，而不是“组合要付多少”。

| 案例 | 架构 | 组件包 | 被复用 | 变体 |
| :--- | :--- | ---: | ---: | ---: |
| Llama-1B decode megakernel | `sm103` | 10 | 8 | 2 |
| MoE 训练 megakernel | `sm103` | 6 | 6 | 24 |
| Attention | `sm90` | 14 | 8 | 7 |
| GEMM + SwiGLU | `sm90` / `sm103` / `sm120` | 9 | 3 | 6 |

### 跨库 megakernel

他们把 Hazy Research 的 Llama-1B decode megakernel（ThunderKittens 写的）拆成九个 package：一个指令解释器加八个算子，五个 role。解释器在一个 persistent kernel 里跑 16 层的预计算指令流；算子在自己的指令 phase 内认领 shared memory 页、semaphore 和 scratch，于是不同算子复用同一个池子。

一个有意思的插曲：原版 megakernel 过不了他们的正确性检查——split-KV 路径会死锁或者从不写部分输出，racecheck 还报出了 warp 之间交接 shared memory 的竞争。他们在每个镜像里（包括单体版）修了五个上游缺陷，所以后面的比较都是“修好之后”的比较。

全部用 ThunderKittens 时，链接版在五个配置上达到单编译单元对照的 0.9995–1.0002× 吞吐，是修好后的单体版的 1.01–1.10×。

然后他们只把 attention-partial 这一个 package 换成 CUTLASS/CuTe 版本：四个 warp 处理不相交的 KV tile，用 warp 级 MMA 算，再在组件内部合并归一化的部分结果。其他八个 package、指令流、编译好的 host provider 都不动——CuTe 包满足同样的契约，所以解释器不需要任何修改。

结果（B300，1155 MHz，ms/token）：

| 上下文 | 纯 TK | TK + CuTe | 加速 |
| :--- | ---: | ---: | ---: |
| 16 | 0.713 | 0.752 | 0.95× |
| 1024（带 split-KV reduction） | 1.151 | 0.939 | 1.22× |
| 4096 | 3.516 | 1.630 | 2.16× |

16 token 时慢 5%，原因写得很直白：四个 warp 里只有一个有活干，组件固定的拷贝与合并开销占了主导。混合版仍在自己单编译单元对照的 0.9994–0.9998×，sanitizer 零发现。这也提醒一句：**2.16× 是换用更合适的算子带来的收益，不是链接的收益**，链接本身是 1.000× 那一档。

### 四个 owner 的 MoE 训练 megakernel

第二个案例是 Mixture-of-Kittens，一个确定性的 expert-parallel MoE 训练 megakernel。它被拆给四个 owner：framework（任务解释器、cluster launch control、launch 几何），以及通信（NVLink 上的 dispatch 和 combine）、grouped GEMM、SwiGLU 三个算子 owner。每个 owner 用 caller 借出的 buffer 在自己的编译产物里构造描述符；别人要依赖的 tile 几何是声明出来的事实；owner 之间的同步走 counter channel，其中一些有三个生产者。

framework 给 forward、backward、recompute 三个 pass 各出一个解释器 package；每个算子 owner 只发一个 package，它的 export 服务所有 pass 和所有变体（BF16 / MXFP8、单卡 / 四卡、clamp / 不 clamp）。六个 package 链进全部 24 个 kernel。

四卡 B300、每 rank 2048 / 8192 token 时，12 个计时 kernel 达到单编译单元对照的 0.999–1.001×，达到原版单体 MoK 整步的 0.989–1.014×。他们还让一个独立的 activation owner 在**不含其他 owner 任何实现源码**的源码树里，按同样的契约实现 GPT-OSS 版 SwiGLU：链进全部 24 个 kernel，其他 owner 的 package 和 provider 不动，每 rank 48 项数值与逐位检查全过。2.4 节那五个“按符号链接会挂或漏写”的单常量升级，按契约链接全部正确重链；七个不兼容组合（比如在 MXFP8 kernel 里放 BF16-only 的通信包）各自被一条具名诊断拒绝。

### 一个 attention core，七个变体

第三个案例最接近“组件复用”的理想形态。他们从 FlashAttention-3 派生出一个 Hopper 上的 attention core（BF16，head dim 128），把它的 package **冻结**，然后七个组合各自只增加或替换至多两个组件：

- sliding window 和 softcapping 是 mask 和 score 两个 callable；
- Phi-3 的 block-sparse attention 和 MiniMax 稀疏 attention（MSA）换掉 load 组件，只访问被选中的 key block；
- context-parallel 换掉 load 组件，再加一个通信组件：把 KV block 推给其他 GPU，每块释放一个 readiness counter。因为每个 query tile 在一个 kernel 内部就访问完自己所有的 KV block，**不需要任何输出修正**。

所有 17 个计时的单卡配置达到单编译单元对照的 0.9991–1.0005×，CP 在两卡和四卡上分别是 0.9997× 和 0.9992×。

和公开实现比（H200，1230 MHz，链接版 / 参考实现的吞吐比）：

| 工作负载 | 参考 | 链接版 / 参考 |
| :--- | :--- | :--- |
| Qwen3-8B causal，8K–32K | FA3 | 0.96–0.99 |
| Mistral / Gemma sliding window | FA3 | 0.84–0.94 |
| Gemma2 / Grok softcap | FA3 | 0.92–0.94 |
| Llama 4 chunked，32K–128K | FA3 | 1.01–1.02 |
| Phi-3 block-sparse，32K–128K | 最好的专家实现 | 2.26–2.73 |
| MSA，prefix 选择 | vLLM Triton（两种模式共用的参考） | 4.47–4.52 |
| MSA，均匀选择 | vLLM Triton（两种模式共用的参考） | 1.62–1.63 |

稀疏变体收益最大，因为它们的 load 组件会跳过没被选中的块。MSA 比 vLLM 的专用 kernel 快 1.6–4.5×，比 FlexAttention 快最多 38×。FA4 在 query tile 上做选择掩码，被选中的块是连续前缀时它快 5–6%，但块分散时慢 9–38×——这是“一个 kernel 服务所有选择模式”的代价。

CP 那边，32K token 的 causal prefill 在两卡上比单卡快 1.95×、四卡上快 3.67×（理想值的 97.6% 和 91.7%）。对比 Transformer Engine 的 context parallelism（用对方更快的传输、并把 forward 捕获进 CUDA graph），PipeLink 版在 8K 和 32K 上快 1.20–1.31×；两边各用自己的本地 attention kernel，所以比较覆盖的是完整实现。

### GEMM + SwiGLU：一个消费者，三代 GPU

最后一个案例把 Qwen3-32B 的 MLP up-projection 和 SwiGLU 融在一起：$X[M, 5120] \times W[51200, 5120]$，输出 BF16 的 $[M, 25600]$，M 取 16384 和 4096，FP32 accumulator 不取整一路传到 activation。

每个架构有自己的 GEMM producer：B300 上是一个 2-SM 的 tensor core GEMM，accumulator 留在 tensor memory；H200 上是三 warpgroup 的 WGMMA GEMM；RTX 5090 上是多级 `mma.sync` GEMM，后两者把 accumulator 留在寄存器。两个消费者（一个 CUDA、一个 TileLang）按 32 列一块 import accumulator，producer 都是**不带源码**链进来的。

生成的搬运路线正好演示了那套阶梯：B300 上链接器把 accumulator 从 tensor memory 直接读进每个消费者线程的寄存器；H200 和 RTX 5090 上，CUDA 消费者走 16 KiB shared memory 中转，而特化过的 TileLang 消费者拿到直接寄存器拷贝。H200 上这条直接路线让 TileLang kernel 比 CUDA 版快 5%（M=16384 时 14.35 ms 对 15.10 ms）。

和基线比（M=16384，CUDA / TileLang 两个消费者）：

| 基线 | B300 | H200 | RTX 5090 |
| :--- | :--- | :--- | :--- |
| cuBLAS + `torch.compile` 的 SwiGLU | 1.20 / 1.20 | 1.03 / 1.08 | 1.03 / 1.01 |
| cuBLAS + Liger 的 SwiGLU | 1.34 / 1.34 | 1.12 / 1.18 | 1.08 / 1.05 |
| 同一个 GEMM，不融合 | 1.11 / 1.10 | 1.02 / 1.07 | 1.03 / 1.01 |
| 手工融合的原生 kernel | 0.99 / 0.98 | 0.94 / 0.99 | 1.01 / 0.99 |

12 个配置都落在单编译单元对照的 0.9987–1.0015×。注意最后一行：相对手工融合的原生实现是 0.94–1.01×，也就是说“能链起来”大致相当于“手工融了”，并不额外送性能。

## 怎么上手

仓库是 [PKU-DAIR/PipeLink](https://github.com/PKU-DAIR/PipeLink)，Apache-2.0。要求 Linux x86_64、CUDA 13.3（`nvcc`、nvJitLink、`cuobjdump`）、C++17 宿主编译器、Python 3 和 `uv`。**打包和链接不需要 GPU**，只跑 kernel 才需要 Hopper、数据中心 Blackwell 或消费级 Blackwell。

```bash
git clone --recursive https://github.com/PKU-DAIR/PipeLink.git
cd PipeLink
cp .env.example .env       # 填 PIPELINK_CUDA_HOME、PIPELINK_SCRATCH_ROOT、PIPELINK_GPUS
source tools/env.sh
uv run --project pipec --locked pipec2 --help
```

`pipec2` 的子命令是 `pack`、`inspect`、`link`、`launch`、`clauses`。最小流程是每个 translation unit 各打一个包，再用一份 pipeline manifest 链起来：

```bash
export PROFILE=sm_103a      # 另有 sm_90a、sm_100a、sm_120a、sm_121a

uv run --project pipec --locked pipec2 pack \
  examples/hello-callable/src/provider.cu --profile "$PROFILE" -o build/provider.plink
uv run --project pipec --locked pipec2 pack \
  examples/hello-callable/src/caller.cu   --profile "$PROFILE" -o build/caller.plink

uv run --project pipec --locked pipec2 link build/pipeline.json --out-dir build/link
```

`.plink` 里是一个 manifest 加一个 `stage.o`，这就是链接的全部输入，源码不是。`pipeline.json` 描述 stages、handles、phases、domains 和 topology（要自己写，T0 给了从两个 `.plink` 生成它的完整脚本）；`link` 产出 `pipeline.cubin`、`link.json` 和生成的源码，`link.json` 里带着 cubin 需要的启动配置（线程数、cluster、shared memory 需求），所以宿主不需要手写 launcher。

官方教程按机制排，T0 和 T3 不需要 GPU：

| 教程 | 内容 | 需要 GPU |
| :--- | :--- | :--- |
| T0 | 安装、pack、link、读 `link.json` | 否 |
| T1 | callable seam：`PL_EXPORT` / `PL_CALL` | 是 |
| T2 | mbarrier ring 流水线，ring 深度在链接时定 | 是 |
| T3 | 换一个 stage、故意破坏契约，看六条具名拒绝 | 否 |
| T4 | 让链接器生成数据通路（CODA 的 SwiGLU 消费者） | 是 |
| T5 | 无源码交付一个 stage：host ABI v3、session、CUDA graph | 是（完整走查要 Hopper） |
| T6 | 用 TileLang 写 stage，对着不透明的 CUDA 链接 | 是 |
| T7 | 把通信 stage 链进计算 kernel，跨 GPU overlap | 2–4 张 Hopper |

## 边界与坑

论文和文档都把边界写得很清楚，这些决定了它适不适合你的场景：

- **契约只检查声明，不检查实现**。一个违反自己契约的组件（比如越界碰了自己没 claim 的 shared memory）仍然会产出错误的 kernel。打包时的 attestation 也只保证 manifest 和符号表对得上，说谎的 manifest 在 link 阶段发现不了（`inspect` 校验 payload digest 是可选动作）。所以数值对拍和 Compute Sanitizer 仍是每个组合的必修课，而链接器检查的只是“两端声明是否自洽”。
- **生成的代码只在 CTA 内搬数据**。不做算术、不做精度转换；跨 CTA 或跨 GPU 的流量写在组件里，链接器只检查并绑定它用的 counter、信号和 peer / multicast 内存。一个 tensor import 一次只交接一代 export，要让 producer 持续产出，得自己在组件里写多槽 ring。
- **它不加速组件**。PipeLink 保留每个 owner 的调度，所以组合的性能就是各组件的性能。稠密 attention 变体与 FA3 的 0.96–0.99 差距来自他们自己的 core，不是链接。
- **已有 kernel 要移植**。把现成实现改成声明式接口是实打实的工作量，四 owner 的 MoE megakernel 就是这条路的样子。
- **一个包绑定一个架构、一个工具链**，因为 LTO IR 不跨 CUDA 大版本。选择 profile 也不会翻译架构相关指令：`hopper-gemm-split` 的 WGMMA 在 `sm_103a` 上打包直接 `P-COMPILE`；TMEM claim 在 `sm_90a` 和 `sm_120a` 上被 `L-HANDLE` 拒绝；CLC 调度和 `cta_group: 2` 在 `sm_90a` / `sm_120a` 上被 `L-TOPO` 拒绝。
- **读性能数字要看清对象**。2.16× 是“把 attention partial 换成 CuTe 实现”的结果，链接本身的开销是 0.2% 以内；16 token 上下文的 0.95× 则是组件在这个 shape 下的固定开销——正是论文自己列出来的反例。

## 和这个博客其他文章的关系

前面几篇讲的是“一个 kernel 怎么写”：CUDA 手册讲执行模型和内存层级，Triton 手册讲 tile 级编程和编译链路，TileLang 手册讲显式 tile 和多后端；DeepGEMM 讲一个生产库已经把 tile、流水线、scale 布局和 MoE 分组固定成了什么样子。

PipeLink 讲的是再上面一层：**当这些 kernel 的组成部分由不同团队、不同库分别编译时，怎么把它们拼成一个 kernel 而不付边界代价**。它和 CUTLASS 这类模板库的分工也在这里——模板库在“一个开发者拥有的一个源程序”里组合，PipeLink 在“多个 owner 的不透明二进制”之间组合，而且不重排别人的调度。

如果你的场景是：同一个 attention core 要支撑稀疏、滑窗、softcap、context-parallel 等一堆变体；或者一个 megakernel 里，通信、GEMM、激活分别由不同团队维护；或者你想在闭源交付一个 kernel 组件——那这套“契约 + 链接期放置”的思路值得细看，哪怕最后不用它的工具链。反过来，如果你只是想写快一个独立 kernel，CUDA / Triton / TileLang 那三篇更直接。

## 参考

- [PipeLink: Contract-Guided Linking of GPU Kernel Components](https://arxiv.org/abs/2610.37200)：arXiv 2610.37200，cs.DC；[HTML 全文](https://arxiv.org/html/2610.37200v1)
- [PKU-DAIR/PipeLink](https://github.com/PKU-DAIR/PipeLink)：编译器、package 格式、`pipec2` CLI、示例与案例复现脚本
- [文档索引](https://github.com/PKU-DAIR/PipeLink/blob/main/docs/index.md)：stage 概念、信任模型、教程阶梯与 profile 覆盖表
- [T0：pack、link、读 link.json](https://github.com/PKU-DAIR/PipeLink/blob/main/docs/tutorials/T0-pack-link.md)：不需要 GPU 的最小流程
- 本分类的相关长文：[CUDA 手册](/posts/kernel/cuda/)、[Triton 手册](/posts/kernel/triton/)、[TileLang 手册](/posts/kernel/tilelang/)、[DeepGEMM 内核库](/posts/kernel/deepgemm/)
