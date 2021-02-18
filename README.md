# JavaScript Algorithms and Data Structures

[![CI](https://github.com/trekhleb/javascript-algorithms/workflows/CI/badge.svg)](https://github.com/trekhleb/javascript-algorithms/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/trekhleb/javascript-algorithms/branch/master/graph/badge.svg)](https://codecov.io/gh/trekhleb/javascript-algorithms)

This repository contains JavaScript based examples of many
popular algorithms and data structures.

Each algorithm and data structure has its own separate README
with related explanations and links for further reading (including ones
to YouTube videos).

_Read this in other languages:_
[_简体中文_](README.zh-CN.md),
[_繁體中文_](README.zh-TW.md),
[_한국어_](README.ko-KR.md),
[_日本語_](README.ja-JP.md),
[_Polski_](README.pl-PL.md),
[_Français_](README.fr-FR.md),
[_Español_](README.es-ES.md),
[_Português_](README.pt-BR.md),
[_Русский_](README.ru-RU.md),
[_Türk_](README.tr-TR.md),
[_Italiana_](README.it-IT.md),
[_Bahasa Indonesia_](README.id-ID.md),
[_Українська_](README.uk-UA.md),
[_Arabic_](README.ar-AR.md)

*☝ Note that this project is meant to be used for learning and researching purposes
only, and it is **not** meant to be used for production.*

## Data Structures

A data structure is a particular way of organizing and storing data in a computer so that it can
be accessed and modified efficiently. More precisely, a data structure is a collection of data
values, the relationships among them, and the functions or operations that can be applied to
the data.

数据结构是计算机中组织和存储数据的一种特殊方式，因此我们需要能高效的访问和修改数据;更重要的是，数据结构是关于数据值，数据之间的关系等的一个集合。

`B` - Beginner, `A` - Advanced

* `B` [链表 Linked List](src/data-structures/linked-list)
* `B` [双向链表 Doubly Linked List](src/data-structures/doubly-linked-list)
* `B` [队列 Queue](src/data-structures/queue)
* `B` [堆栈 Stack](src/data-structures/stack)
* `B` [哈希表 Hash Table](src/data-structures/hash-table)
* `B` [堆 Heap](src/data-structures/heap) - max and min heap versions
* `B` [优先队列 Priority Queue](src/data-structures/priority-queue)
* `A` [字典树 Trie](src/data-structures/trie)
* `A` [树 Tree](src/data-structures/tree)
  * `A` [二叉查找树 Binary Search Tree](src/data-structures/tree/binary-search-tree)
  * `A` [平衡二叉搜索树 ](src/data-structures/tree/avl-tree)
  * `A` [红黑树 Red-Black Tree](src/data-structures/tree/red-black-tree)
  * `A` [线段树 Segment Tree](src/data-structures/tree/segment-tree) - with min/max/sum range queries examples
  * `A` [树状数组 Fenwick Tree](src/data-structures/tree/fenwick-tree) (Binary Indexed Tree)
* `A` [图 Graph](src/data-structures/graph) (both directed and undirected)
* `A` [并查集 Disjoint Set](src/data-structures/disjoint-set)
* `A` [布隆过滤器 Bloom Filter](src/data-structures/bloom-filter)

## Algorithms

An algorithm is an unambiguous specification of how to solve a class of problems. It is
a set of rules that precisely define a sequence of operations.

语法说明了如何处理一系列这样的问题。这个集合中特别定义了一系列的操作方法。

`B` - Beginner, `A` - Advanced

### Algorithms by Topic

* **Math**
  * `B` [位运算 Bit Manipulation](src/algorithms/math/bits) - set/get/update/clear bits, multiplication/division by two, make negative etc.
  * `B` [阶乘 Factorial](src/algorithms/math/factorial)
  * `B` [斐波那契树列 Fibonacci Number](src/algorithms/math/fibonacci) - classic and closed-form versions
  * `B` [质因子 Prime Factors](src/algorithms/math/prime-factors) - finding prime factors and counting them using Hardy-Ramanujan's theorem
  * `B` [素数测试 Primality Test](src/algorithms/math/primality-test) (trial division method)
  * `B` [欧几里德算法 Euclidean Algorithm](src/algorithms/math/euclidean-algorithm) - calculate the Greatest Common Divisor (GCD)
  * `B` [最小共倍数 Least Common Multiple](src/algorithms/math/least-common-multiple) (LCM)
  * `B` [质数筛选法 Sieve of Eratosthenes](src/algorithms/math/sieve-of-eratosthenes) - finding all prime numbers up to any given limit
  * `B` [判断2的次幂 Is Power of Two](src/algorithms/math/is-power-of-two) - check if the number is power of two (naive and bitwise algorithms)
  * `B` [杨辉三角 Pascal's Triangle](src/algorithms/math/pascal-triangle)
  * `B` [Complex Number](src/algorithms/math/complex-number) - complex numbers and basic operations with them
  * `B` [弧度与角度 Radian & Degree](src/algorithms/math/radian) - radians to degree and backwards conversion
  * `B` [快速幂 Fast Powering](src/algorithms/math/fast-powering)
  * `B` [快速计算多项式-霍纳方法 Horner's method](src/algorithms/math/horner-method) - polynomial evaluation
  * `B` [矩阵 Matrices](src/algorithms/math/matrix) - matrices and basic matrix operations (multiplication, transposition, etc.)
  * `B` [欧几里的距离 Euclidean Distance](src/algorithms/math/euclidean-distance) - distance between two points/vectors/matrices
  * `A` [整数划分 Integer Partition](src/algorithms/math/integer-partition)
  * `A` [平方根 Square Root](src/algorithms/math/square-root) - Newton's method
  * `A` [刘辉分圆法 Liu Hui π Algorithm](src/algorithms/math/liu-hui) - approximate π calculations based on N-gons
  * `A` [离散傅里叶变换 Discrete Fourier Transform](src/algorithms/math/fourier-transform) - decompose a function of time (a signal) into the frequencies that make it up
* **Sets 集合**
  * `B` [笛卡尔乘积 Cartesian Product](src/algorithms/sets/cartesian-product) - product of multiple sets
  * `B` [洗牌算法 Fisher–Yates Shuffle](src/algorithms/sets/fisher-yates) - random permutation of a finite sequence
  * `A` [幂集 Power Set](src/algorithms/sets/power-set) - all subsets of a set (bitwise and backtracking solutions)
  * `A` [全排列 Permutations](src/algorithms/sets/permutations) (with and without repetitions)
  * `A` [排列组合 Combinations](src/algorithms/sets/combinations) (with and without repetitions)
  * `A` [最长公共子序列 Longest Common Subsequence](src/algorithms/sets/longest-common-subsequence) (LCS)
  * `A` [最长递增子串 Longest Increasing Subsequence](src/algorithms/sets/longest-increasing-subsequence)
  * `A` [最短公共子序列 Shortest Common Supersequence](src/algorithms/sets/shortest-common-supersequence) (SCS)
  * `A` [背包问题 Knapsack Problem](src/algorithms/sets/knapsack-problem) - "0/1" and "Unbound" ones
  * `A` [最大子数组 Maximum Subarray](src/algorithms/sets/maximum-subarray) - "Brute Force" and "Dynamic Programming" (Kadane's) versions
  * `A` [组合之和 Combination Sum](src/algorithms/sets/combination-sum) - find all combinations that form specific sum
* **Strings 字符串**
  * `B` [汉明距离 Hamming Distance](src/algorithms/string/hamming-distance) - number of positions at which the symbols are different
  * `A` [编辑距离 Levenshtein Distance](src/algorithms/string/levenshtein-distance) - minimum edit distance between two sequences
  * `A` [KMP算法 Knuth–Morris–Pratt Algorithm](src/algorithms/string/knuth-morris-pratt) (KMP Algorithm) - substring search (pattern matching)
  * `A` [Z Algorithm](src/algorithms/string/z-algorithm) - substring search (pattern matching)
  * `A` [字符串匹配 Rabin Karp Algorithm](src/algorithms/string/rabin-karp) - substring search
  * `A` [最长的公共子串 Longest Common Substring](src/algorithms/string/longest-common-substring)
  * `A` [正则表达式匹配 Regular Expression Matching](src/algorithms/string/regular-expression-matching)
* **Searches 搜索方法**
  * `B` [线性搜索 Linear Search](src/algorithms/search/linear-search)
  * `B` [跳跃搜索 Jump Search](src/algorithms/search/jump-search) (or Block Search) - search in sorted array
  * `B` [二分法搜索 Binary Search](src/algorithms/search/binary-search) - search in sorted array
  * `B` [插值查找 Interpolation Search](src/algorithms/search/interpolation-search) - search in uniformly distributed sorted array
* **Sorting 排序**
  * `B` [冒泡排序 Bubble Sort](src/algorithms/sorting/bubble-sort)
  * `B` [选择排序 Selection Sort](src/algorithms/sorting/selection-sort)
  * `B` [插值排序 Insertion Sort](src/algorithms/sorting/insertion-sort)
  * `B` [堆排序 Heap Sort](src/algorithms/sorting/heap-sort)
  * `B` [归并排序 Merge Sort](src/algorithms/sorting/merge-sort)
  * `B` [快速排序 Quicksort](src/algorithms/sorting/quick-sort) - in-place and non-in-place implementations
  * `B` [希尔排序 Shellsort](src/algorithms/sorting/shell-sort)
  * `B` [计数排序 Counting Sort](src/algorithms/sorting/counting-sort)
  * `B` [基数排序 Radix Sort](src/algorithms/sorting/radix-sort)
* **Linked Lists 链表**
  * `B` [垂直遍历 Straight Traversal](src/algorithms/linked-list/traversal)
  * `B` [反向遍历 Reverse Traversal](src/algorithms/linked-list/reverse-traversal)
* **Trees 树**
  * `B` [深度优先搜索 Depth-First Search](src/algorithms/tree/depth-first-search) (DFS)
  * `B` [广度优先搜索 Breadth-First Search](src/algorithms/tree/breadth-first-search) (BFS)
* **Graphs 图**
  * `B` [深度优先搜索 Depth-First Search](src/algorithms/graph/depth-first-search) (DFS)
  * `B` [广度优先搜索 Breadth-First Search](src/algorithms/graph/breadth-first-search) (BFS)
  * `B` [克鲁斯卡尔算法 Kruskal’s Algorithm](src/algorithms/graph/kruskal) - finding Minimum Spanning Tree (MST) for weighted undirected graph
  * `A` [最短路径算法 Dijkstra Algorithm](src/algorithms/graph/dijkstra) - finding the shortest paths to all graph vertices from single vertex
  * `A` [贝尔曼-福特算法 Bellman-Ford Algorithm](src/algorithms/graph/bellman-ford) - finding the shortest paths to all graph vertices from single vertex
  * `A` [多源最短路径 Floyd-Warshall Algorithm](src/algorithms/graph/floyd-warshall) - find the shortest paths between all pairs of vertices
  * `A` [Detect Cycle](src/algorithms/graph/detect-cycle) - for both directed and undirected graphs (DFS and Disjoint Set based versions)
  * `A` [Prim’s Algorithm](src/algorithms/graph/prim) - finding Minimum Spanning Tree (MST) for weighted undirected graph
  * `A` [拓扑排序 Topological Sorting](src/algorithms/graph/topological-sorting) - DFS method
  * `A` [关节点 Articulation Points](src/algorithms/graph/articulation-points) - Tarjan's algorithm (DFS based)
  * `A` [桥 Bridges](src/algorithms/graph/bridges) - DFS based algorithm
  * `A` [欧拉路径和欧拉回环 Eulerian Path and Eulerian Circuit](src/algorithms/graph/eulerian-path) - Fleury's algorithm - Visit every edge exactly once
  * `A` [汉密尔顿圆 Hamiltonian Cycle](src/algorithms/graph/hamiltonian-cycle) - Visit every vertex exactly once
  * `A` [强连通分量 Strongly Connected Components](src/algorithms/graph/strongly-connected-components) - Kosaraju's algorithm
  * `A` [旅行商问题 Travelling Salesman Problem](src/algorithms/graph/travelling-salesman) - shortest possible route that visits each city and returns to the origin city
* **Cryptography 密码学**
  * `B` [多项式哈希 Polynomial Hash](src/algorithms/cryptography/polynomial-hash) - rolling hash function based on polynomial
  * `B` [栅栏密码 Rail Fence Cipher](src/algorithms/cryptography/rail-fence-cipher) - a transposition cipher algorithm for encoding messages
  * `B` [凯撒密码 Caesar Cipher](src/algorithms/cryptography/caesar-cipher) - simple substitution cipher
  * `B` [希尔密码 Hill Cipher](src/algorithms/cryptography/hill-cipher) - substitution cipher based on linear algebra
* **Machine Learning 机器学习**
  * `B` [微型神经元 NanoNeuron](https://github.com/trekhleb/nano-neuron) - 7 simple JS functions that illustrate how machines can actually learn (forward/backward propagation)
  * `B` [k-NN](src/algorithms/ml/knn) - k-nearest neighbors classification algorithm
  * `B` [K均值聚类算法 k-Means](src/algorithms/ml/k-means) - k-Means clustering algorithm
* **Uncategorized 未分类的**
  * `B` [汉诺塔 Tower of Hanoi](src/algorithms/uncategorized/hanoi-tower)
  * `B` [方阵旋转 Square Matrix Rotation](src/algorithms/uncategorized/square-matrix-rotation) - in-place algorithm
  * `B` [跳跃游戏 Jump Game](src/algorithms/uncategorized/jump-game) - backtracking, dynamic programming (top-down + bottom-up) and greedy examples
  * `B` [唯一路径 Unique Paths](src/algorithms/uncategorized/unique-paths) - backtracking, dynamic programming and Pascal's Triangle based examples
  * `B` [雨水收集问题 Rain Terraces](src/algorithms/uncategorized/rain-terraces) - trapping rain water problem (dynamic programming and brute force versions)
  * `B` [递归楼梯问题 Recursive Staircase](src/algorithms/uncategorized/recursive-staircase) - count the number of ways to reach to the top (4 solutions)
  * `B` [买卖股票的最佳时机 Best Time To Buy Sell Stocks](src/algorithms/uncategorized/best-time-to-buy-sell-stocks) - divide and conquer and one-pass examples
  * `A` [N皇后问题 N-Queens Problem](src/algorithms/uncategorized/n-queens)
  * `A` [骑士巡游问题 Knight's Tour](src/algorithms/uncategorized/knight-tour)

### Algorithms by Paradigm 典型算法实例

An algorithmic paradigm is a generic method or approach which underlies the design of a class
of algorithms. It is an abstraction higher than the notion of an algorithm, just as an
algorithm is an abstraction higher than a computer program.

* **Brute Force 暴力搜索** - look at all the possibilities and selects the best solution
  * `B` [线性搜索 Linear Search](src/algorithms/search/linear-search)
  * `B` [雨水收集问题 Rain Terraces](src/algorithms/uncategorized/rain-terraces) - trapping rain water problem
  * `B` [递归楼梯 Recursive Staircase](src/algorithms/uncategorized/recursive-staircase) - count the number of ways to reach to the top
  * `A` [最大连续子数组 Maximum Subarray](src/algorithms/sets/maximum-subarray)
  * `A` [旅行商问题 Travelling Salesman Problem](src/algorithms/graph/travelling-salesman) - shortest possible route that visits each city and returns to the origin city
  * `A` [离散傅里叶变换 Discrete Fourier Transform](src/algorithms/math/fourier-transform) - decompose a function of time (a signal) into the frequencies that make it up
* **Greedy 贪心算法** - choose the best option at the current time, without any consideration for the future
  * `B` [跳跃游戏 Jump Game](src/algorithms/uncategorized/jump-game)
  * `A` [无约束背包问题 Unbound Knapsack Problem](src/algorithms/sets/knapsack-problem)
  * `A` [最短距离算法 Dijkstra Algorithm](src/algorithms/graph/dijkstra) - finding the shortest path to all graph vertices
  * `A` [Prim’s Algorithm](src/algorithms/graph/prim) - finding Minimum Spanning Tree (MST) for weighted undirected graph
  * `A` [Kruskal’s Algorithm](src/algorithms/graph/kruskal) - finding Minimum Spanning Tree (MST) for weighted undirected graph
* **Divide and Conquer 分治法** - divide the problem into smaller parts and then solve those parts
  * `B` [Binary Search](src/algorithms/search/binary-search)
  * `B` [Tower of Hanoi](src/algorithms/uncategorized/hanoi-tower)
  * `B` [杨辉三角 Pascal's Triangle](src/algorithms/math/pascal-triangle)
  * `B` [Euclidean Algorithm](src/algorithms/math/euclidean-algorithm) - calculate the Greatest Common Divisor (GCD)
  * `B` [Merge Sort](src/algorithms/sorting/merge-sort)
  * `B` [Quicksort](src/algorithms/sorting/quick-sort)
  * `B` [Tree Depth-First Search](src/algorithms/tree/depth-first-search) (DFS)
  * `B` [Graph Depth-First Search](src/algorithms/graph/depth-first-search) (DFS)
  * `B` [Matrices](src/algorithms/math/matrix) - generating and traversing the matrices of different shapes
  * `B` [Jump Game](src/algorithms/uncategorized/jump-game)
  * `B` [Fast Powering](src/algorithms/math/fast-powering)
  * `B` [Best Time To Buy Sell Stocks](src/algorithms/uncategorized/best-time-to-buy-sell-stocks) - divide and conquer and one-pass examples
  * `A` [全排列 Permutations](src/algorithms/sets/permutations) (with and without repetitions)
  * `A` [排列组合 Combinations](src/algorithms/sets/combinations) (with and without repetitions)
* **Dynamic Programming 动态规划** - build up a solution using previously found sub-solutions
  * `B` [斐波那契数列 Fibonacci Number](src/algorithms/math/fibonacci)
  * `B` [Jump Game](src/algorithms/uncategorized/jump-game)
  * `B` [Unique Paths](src/algorithms/uncategorized/unique-paths)
  * `B` [Rain Terraces](src/algorithms/uncategorized/rain-terraces) - trapping rain water problem
  * `B` [Recursive Staircase](src/algorithms/uncategorized/recursive-staircase) - count the number of ways to reach to the top
  * `A` [Levenshtein Distance](src/algorithms/string/levenshtein-distance) - minimum edit distance between two sequences
  * `A` [Longest Common Subsequence](src/algorithms/sets/longest-common-subsequence) (LCS)
  * `A` [Longest Common Substring](src/algorithms/string/longest-common-substring)
  * `A` [Longest Increasing Subsequence](src/algorithms/sets/longest-increasing-subsequence)
  * `A` [Shortest Common Supersequence](src/algorithms/sets/shortest-common-supersequence)
  * `A` [0/1背包问题 0/1 Knapsack Problem](src/algorithms/sets/knapsack-problem)
  * `A` [Integer Partition](src/algorithms/math/integer-partition)
  * `A` [Maximum Subarray](src/algorithms/sets/maximum-subarray)
  * `A` [Bellman-Ford Algorithm](src/algorithms/graph/bellman-ford) - finding the shortest path to all graph vertices
  * `A` [Floyd-Warshall Algorithm](src/algorithms/graph/floyd-warshall) - find the shortest paths between all pairs of vertices
  * `A` [Regular Expression Matching](src/algorithms/string/regular-expression-matching)
* **Backtracking 回溯法** - similarly to brute force, try to generate all possible solutions, but each time you generate next solution you test
if it satisfies all conditions, and only then continue generating subsequent solutions. Otherwise, backtrack, and go on a
different path of finding a solution. Normally the DFS traversal of state-space is being used.
  * `B` [Jump Game](src/algorithms/uncategorized/jump-game)
  * `B` [Unique Paths](src/algorithms/uncategorized/unique-paths)
  * `B` [Power Set](src/algorithms/sets/power-set) - all subsets of a set
  * `A` [Hamiltonian Cycle](src/algorithms/graph/hamiltonian-cycle) - Visit every vertex exactly once
  * `A` [N-Queens Problem](src/algorithms/uncategorized/n-queens)
  * `A` [Knight's Tour](src/algorithms/uncategorized/knight-tour)
  * `A` [Combination Sum](src/algorithms/sets/combination-sum) - find all combinations that form specific sum
* **Branch & Bound** - remember the lowest-cost solution found at each stage of the backtracking
search, and use the cost of the lowest-cost solution found so far as a lower bound on the cost of
a least-cost solution to the problem, in order to discard partial solutions with costs larger than the
lowest-cost solution found so far. Normally BFS traversal in combination with DFS traversal of state-space
tree is being used.

## How to use this repository

**Install all dependencies**
```
npm install
```

**Run ESLint**

You may want to run it to check code quality.

```
npm run lint
```

**Run all tests**
```
npm test
```

**Run tests by name**
```
npm test -- 'LinkedList'
```

**Playground**

You may play with data-structures and algorithms in `./src/playground/playground.js` file and write
tests for it in `./src/playground/__test__/playground.test.js`.

Then just simply run the following command to test if your playground code works as expected:

```
npm test -- 'playground'
```

## Useful Information

### References

[▶ Data Structures and Algorithms on YouTube](https://www.youtube.com/playlist?list=PLLXdhg_r2hKA7DPDsunoDZ-Z769jWn4R8)

### Big O Notation

*Big O notation* is used to classify algorithms according to how their running time or space requirements grow as the input size grows.
On the chart below you may find most common orders of growth of algorithms specified in Big O notation.

![Big O graphs](./assets/big-o-graph.png)

Source: [Big O Cheat Sheet](http://bigocheatsheet.com/).

Below is the list of some of the most used Big O notations and their performance comparisons against different sizes of the input data.

| Big O Notation | Computations for 10 elements | Computations for 100 elements | Computations for 1000 elements  |
| -------------- | ---------------------------- | ----------------------------- | ------------------------------- |
| **O(1)**       | 1                            | 1                             | 1                               |
| **O(log N)**   | 3                            | 6                             | 9                               |
| **O(N)**       | 10                           | 100                           | 1000                            |
| **O(N log N)** | 30                           | 600                           | 9000                            |
| **O(N^2)**     | 100                          | 10000                         | 1000000                         |
| **O(2^N)**     | 1024                         | 1.26e+29                      | 1.07e+301                       |
| **O(N!)**      | 3628800                      | 9.3e+157                      | 4.02e+2567                      |

### Data Structure Operations Complexity

| Data Structure          | Access    | Search    | Insertion | Deletion  | Comments  |
| ----------------------- | :-------: | :-------: | :-------: | :-------: | :-------- |
| **Array**               | 1         | n         | n         | n         |           |
| **Stack**               | n         | n         | 1         | 1         |           |
| **Queue**               | n         | n         | 1         | 1         |           |
| **Linked List**         | n         | n         | 1         | n         |           |
| **Hash Table**          | -         | n         | n         | n         | In case of perfect hash function costs would be O(1) |
| **Binary Search Tree**  | n         | n         | n         | n         | In case of balanced tree costs would be O(log(n)) |
| **B-Tree**              | log(n)    | log(n)    | log(n)    | log(n)    |           |
| **Red-Black Tree**      | log(n)    | log(n)    | log(n)    | log(n)    |           |
| **AVL Tree**            | log(n)    | log(n)    | log(n)    | log(n)    |           |
| **Bloom Filter**        | -         | 1         | 1         | -         | False positives are possible while searching |

### Array Sorting Algorithms Complexity

| Name                  | Best            | Average             | Worst               | Memory    | Stable    | Comments  |
| --------------------- | :-------------: | :-----------------: | :-----------------: | :-------: | :-------: | :-------- |
| **Bubble sort**       | n               | n<sup>2</sup>       | n<sup>2</sup>       | 1         | Yes       |           |
| **Insertion sort**    | n               | n<sup>2</sup>       | n<sup>2</sup>       | 1         | Yes       |           |
| **Selection sort**    | n<sup>2</sup>   | n<sup>2</sup>       | n<sup>2</sup>       | 1         | No        |           |
| **Heap sort**         | n&nbsp;log(n)   | n&nbsp;log(n)       | n&nbsp;log(n)       | 1         | No        |           |
| **Merge sort**        | n&nbsp;log(n)   | n&nbsp;log(n)       | n&nbsp;log(n)       | n         | Yes       |           |
| **Quick sort**        | n&nbsp;log(n)   | n&nbsp;log(n)       | n<sup>2</sup>       | log(n)    | No        | Quicksort is usually done in-place with O(log(n)) stack space |
| **Shell sort**        | n&nbsp;log(n)   | depends on gap sequence   | n&nbsp;(log(n))<sup>2</sup>  | 1         | No         |           |
| **Counting sort**     | n + r           | n + r               | n + r               | n + r     | Yes       | r - biggest number in array |
| **Radix sort**        | n * k           | n * k               | n * k               | n + k     | Yes       | k - length of longest key |

## Project Backers

> You may support this project via ❤️️ [GitHub](https://github.com/sponsors/trekhleb) or ❤️️ [Patreon](https://www.patreon.com/trekhleb).

[Folks who are backing this project](https://github.com/trekhleb/javascript-algorithms/blob/master/BACKERS.md) `∑ = 0`
