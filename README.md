
# Introduction
This project inherit from [MMCV](https://github.com/open-mmlab/mmcv/tree/master) for practice cuda and cpp extension in torch.

# Installation
``` shell
pip install -e .
```

# 详细解剖
More details for MMCV build as follow image:
![](resource/MMCV-Build-V1.4.7.png)


## 1.1 Register+Dispatch

- 简化版的代码可以参考c++里面`main执行之前进行操作`章节

``` c++
// 定义一个DeviceRegistry模板类，输入两个参数
// 第一个参数是函数指针，第二个参数是函数（无用）
template <typename F, F f>
class DeviceRegistry;

// 模板类外面套了一层模板，用于可变参数的输入和函数的输入
// 注意：可变参数在中间，后面的参数一定是可推导的类型
template <typename Ret, typename... Args, Ret (*f)(Args...)>
// DeviceRegistry模板类具体化
class DeviceRegistry<Ret (*)(Args...), f> {
 public:
  using FunctionType = Ret (*)(Args...); // 重命名类型
  static const int MAX_DEVICE_TYPES =
      int8_t(at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
	// 将函数地址注册进函数指针中
  void Register(at::DeviceType device, FunctionType function) {
    funcs_[int8_t(device)] = function;
  }
	// 根据type类型查找函数
  FunctionType Find(at::DeviceType device) const {
    return funcs_[int8_t(device)];
  }
	// 注意这里是static数据，返回引用变成全局变量
  static DeviceRegistry& instance() {
    static DeviceRegistry inst;
    return inst;
  }

 private:
  DeviceRegistry() {
    for (size_t i = 0; i < MAX_DEVICE_TYPES; ++i) {
      funcs_[i] = nullptr;
    }
  };
  FunctionType funcs_[MAX_DEVICE_TYPES];
};

```

``` c++
// 获得实例化的类（注意不同的template，实例的对象也是不同的）
define DEVICE_REGISTRY(key) DeviceRegistry<decltype(&(key)), key>::instance()
// 此处是难点
// 1. struct的名字不能重复，因为#define会在编译阶段检查，CPU/GPU使用不同的名字不然重复
// 2. 使用struct的构造函数去调用DEVICE_REGISTRY，因为c++中执行代码必须在函数/类/main中，还有这里一个特殊的构造函数
#define REGISTER_DEVICE_IMPL(key, device, value)           \
  struct key##_##device##_registerer {                     \
    key##_##device##_registerer() {                        \
      DEVICE_REGISTRY(key).Register(at::k##device, value); \
    }                                                      \
  };                                                       \
  static key##_##device##_registerer _##key##_##device##_registerer;

#define DISPATCH_DEVICE_IMPL(key, ...) \
  Dispatch(DEVICE_REGISTRY(key), #key, __VA_ARGS__)

```

``` c++
// cpu的注册，第一个参数是一个函数，有两个作用
// 1. nms_impl作为template的输入（通过decltype解析类型），CPU和GPU使用相同的函数，因为公用一个`static DeviceRegistry inst;`
// 2. nms_impl作用DISPATCH_DEVICE_IMPL的调用
REGISTER_DEVICE_IMPL(nms_impl, CPU, nms_cpu);
REGISTER_DEVICE_IMPL(nms_impl, GPU, nms_gpu);
```



# FAQ
