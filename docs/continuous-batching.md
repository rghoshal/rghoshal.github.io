# Continuous Batching: The Secret Behind High-Throughput LLM Serving
----

# Introduction

In its quest to generate the next "best" token, Large Language Models (LLMs) face a significant challenge: the inference latency can be substantial, especially when generating long sequences of text. This is because the model has to perform a forward pass through the entire network for each token it generates. While this approach is simple and effective for short sequences, it becomes prohibitively expensive for long sequences. To address this, researchers have developed various optimization techniques, one of which is continuous batching.

## The Problem: Static Batching and GPU Underutilization

Static batching is the traditional approach to LLM inference. In static batching, a fixed number of requests are grouped together to form a batch. The model processes the entire batch in a single forward pass. The size of the batch is fixed and cannot be changed during inference. This approach is simple and effective for short sequences, but it becomes prohibitively expensive for long sequences. The reason for this is that the model has to perform a forward pass through the entire network for each token it generates. While this approach is simple and effective for short sequences, it becomes prohibitively expensive for long sequences. To address this, researchers have developed various optimization techniques, one of which is continuous batching.

One of the impacts of statis batching is inefficient or sub-optimized GPU memory utilization. This is because the memory required for each request is determined by the length of the input prompt plus the number of tokens to be generated. However, since the batch size is fixed, the memory required for the batch is determined by the longest sequence in the batch. This means that if the requests in the batch have different lengths, the memory required for the batch will be determined by the longest sequence in the batch. This can lead to a situation where the GPU is not being used to its full potential, as the memory required for the batch is determined by the longest sequence in the batch, even if the other requests in the batch have shorter lengths. This can lead to a situation where the GPU is not being used to its full potential, leading to unintended throttling of the request streams that throng the transformer pipeline looking for next generation sequence.

## The Solution: Continuous Batching
As a solution to the problem mentioned above , a family of algorithms have been designed that allows requests to be processed in a flexible, dynamic manner rather than in rigid, fixed-size batches. In this approach , different requests can have different lengths , and the batch size can change dynamically during inference. This is achieved by keeping track of the memory usage of each request and adjusting the batch size accordingly. The actual memory usage during inference is determined by the sum of the memory usage of all the requests in the batch. As each request generates a new token, the memory usage of that request increases. The batch size is adjusted dynamically to ensure that the total memory usage does not exceed the available memory. This allows the GPU to be used more efficiently, as the memory is not being wasted on requests that are not being processed. In addition to improved memory utilization , continuous batching also allows for lower latency and higher throughput , as the requests are processed in a more efficient manner. 

## Key Optimization Techniques in Continuous Batching

Continuous batching , while conceptually simple , relies on a few key optimization techniques to achieve its performance benefits. These techniques work together to allow for dynamic batch sizing and efficient memory utilization. Some of the key optimization techniques in continuous batching include :

1. **Padding-Free Attention**: In traditional static batching , requests are padded to the length of the longest sequence in the batch. This means that even if a request has a short sequence , it will still occupy memory and computational resources as if it were a long sequence. Padding-free attention eliminates this waste by allowing each request to be processed independently of the other requests in the batch. This is achieved by using a sparse attention mechanism that only attends to the relevant tokens in the sequence. As a result , the memory and computational resources required for each request are determined by the actual length of the sequence , rather than the length of the longest sequence in the batch. 

2. **Dynamic Batch Sizing**: Continuous batching allows the batch size to change dynamically during inference. This is achieved by keeping track of the memory usage of each request and adjusting the batch size accordingly. As each request generates a new token, the memory usage of that request increases. The batch size is adjusted dynamically to ensure that the total memory usage does not exceed the available memory. This allows the GPU to be used more efficiently, as the memory is not being wasted on requests that are not being processed. In addition to improved memory utilization , continuous batching also allows for lower latency and higher throughput , as the requests are processed in a more efficient manner. 

3. **Prefill and Decode**: Continuous batching often involves a two-phase approach to processing requests. In the first phase , called prefill , the input prompts of all the requests in the batch are processed in parallel. This is done to generate the initial key-value (KV) cache for each request. In the second phase , called decode , the requests are processed one token at a time. This is done to generate the output sequences for each request. The two-phase approach allows for efficient use of the GPU, as the prefill phase can be parallelized across all the requests in the batch, while the decode phase can be optimized for single-token generation. 

4. **Speculative Decoding**: Speculative decoding is an optimization technique that can be used to further improve the performance of continuous batching. In speculative decoding , a smaller, faster draft model is used to generate a few candidate tokens in parallel with the main model. These candidate tokens are then verified by the main model in a single forward pass. If the candidate tokens are correct , they are accepted as the output. If the candidate tokens are incorrect , they are discarded and the main model generates the correct tokens. Speculative decoding can significantly reduce the latency and improve the throughput of continuous batching. 

## Benchmarking and Performance Evaluation

To evaluate the performance of continuous batching , it is essential to have a comprehensive benchmarking framework that can measure the key performance indicators (KPIs). These KPIs help in understanding the efficiency of the system under different load conditions. The most important KPIs for evaluating continuous batching include :

1. **Throughput**: This is the measure of the number of requests that can be processed by the system in a given time period. A higher throughput indicates better system performance. In the context of continuous batching , throughput is often measured in tokens per second (TPS) or requests per second (RPS). A good continuous batching system should be able to achieve high throughput even under heavy load.

2. **Latency**: This is the measure of the time it takes for a single request to be processed by the system. Latency is often measured in milliseconds (ms). A lower latency indicates better system performance. In the context of continuous batching , latency is often measured as the time it takes to generate the first token (time-to-first-token) or the time it takes to generate all the tokens (time-to-completion). A good continuous batching system should be able to achieve low latency even under heavy load.

3. **Memory Efficiency**: This is the measure of how efficiently the system uses the available memory. Memory efficiency is often measured as the percentage of memory that is used by the system. A higher memory efficiency indicates better system performance. In the context of continuous batching , memory efficiency is often measured as the percentage of memory that is used by the active requests. A good continuous batching system should be able to achieve high memory efficiency even under heavy load.

## Popular Implementations and Frameworks

Several open-source frameworks and libraries have been developed to implement continuous batching. These frameworks provide optimized implementations of the key optimization techniques discussed above, making it easier for developers to deploy and use continuous batching. Some of the popular implementations include :

1. **vLLM**: vLLM is a high-throughput and memory-efficient LLM inference engine that supports continuous batching. It uses PagedAttention, an algorithm that manages memory more efficiently, and implements continuous batching to maximize GPU utilization. vLLM has become a popular choice for deploying LLMs due to its ease of use and excellent performance.

2. **TensorRT-LLM**: TensorRT-LLM is an open-source library from NVIDIA that is designed for high-performance LLM inference. It supports continuous batching and incorporates various optimizations like quantization and kernel fusion to maximize throughput and minimize latency. It is tightly integrated with NVIDIA's TensorRT ecosystem, providing a robust platform for large-scale deployments.

3. **DeepSpeed-MII**: DeepSpeed-MII (Model Inference and Inference) is part of Microsoft's DeepSpeed optimization library. It is built on top of DeepSpeed's inference optimizations and supports continuous batching. It also incorporates DeepSpeed's features like quantization and ZeRO (Zero Redundancy Optimizer) to further enhance performance. 

4. **Hugging Face Text Generation Inference (TGI)**: TGI is a production-ready inference server for transformer models from Hugging Face. It supports continuous batching and includes features like token streaming and quantization to optimize performance. TGI is widely used for deploying Hugging Face models in production environments.

## Common Challenges and Limitations

While continuous batching offers significant benefits, it also comes with certain challenges and limitations that need to be considered. These challenges can impact the performance of the system and need to be addressed during deployment. Some of the common challenges include :

1. **Complexity**: Continuous batching is more complex to implement than traditional static batching. It requires careful management of memory and computation resources, as well as the implementation of various optimization techniques. This complexity can make it challenging to deploy and maintain continuous batching systems, especially for smaller teams or organizations. 

2. **Memory Overhead**: While continuous batching improves memory efficiency compared to static batching, it still requires significant memory resources. The KV cache, which stores the intermediate activations of the model, can consume a large amount of memory, especially for long sequences. This can be a limitation for deployments with limited memory resources.
