..
    # Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
    # project contributors. See the CHAI LICENSE file for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _user_guide:

**********
User Guide
**********

-----------------------------------
A Portable Pattern for Polymorphism
-----------------------------------

CHAI provides a data structure to help handle cases where it is desirable to call virtual functions on the device. If you only call virtual functions on the host, this pattern is unnecessary. But for those who do want to use virtual functions on the device without a painstaking amount of refactoring, we begin with a short, albeit admittedly contrived example.

.. code-block:: cpp

   class MyBaseClass {
      public:
         MyBaseClass() {}
         virtual ~MyBaseClass() {}
         virtual int getValue() const = 0;
   };

   class MyDerivedClass : public MyBaseClass {
      public:
         MyDerivedClass(int value) : MyBaseClass(), m_value(value) {}
         ~MyDerivedClass() {}
         int getValue() const { return m_value; }

      private:
         int m_value;
   };

   int main(int argc, char** argv) {
      MyBaseClass* myBaseClass = new MyDerivedClass(0);
      myBaseClass->getValue();
      delete myBaseClass;
      return 0;
   }

It is perfectly fine to call `myBaseClass->getValue()` in host code, since `myBaseClass` was created on the host. However, what if you want to call this virtual function on the device?

.. code-block:: cpp

   __global__ void callVirtualFunction(MyBaseClass* myBaseClass) {
      myBaseClass->getValue();
   }

   int main(int argc, char** argv) {
      MyBaseClass* myBaseClass = new MyDerivedClass(0);
      callVirtualFunction<<<1, 1>>>(myBaseClass);
      delete myBaseClass;
      return 0;
   }

At best, calling this code will result in a crash. At worst, it will access garbage and happily continue while giving incorrect results. It is illegal to access host pointers on the device and produces undefined behavior. So what is our next attempt? Why not pass the argument by value rather than by a pointer?

.. code-block:: cpp

   __global__ void callVirtualFunction(MyBaseClass myBaseClass) {
      myBaseClass.getValue();
   }

   int main(int argc, char** argv) {
      MyBaseClass* myBaseClass = new MyDerivedClass(0);
      callVirtualFunction<<<1, 1>>>(*myBaseClass); // This will not compile
      delete myBaseClass;
      return 0;
   }

At first glance, this may seem like it would work, but this is not supported by nvidia: "It is not allowed to pass as an argument to a `__global__` function an object of a class with virtual functions" (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions). Also: "It is not allowed to pass as an argument to a `__global__` function an object of a class derived from virtual base classes" (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-base-classes). You could refactor to use the curiously recurring template pattern, but that would likely require a large development effort and also limits the programming patterns you can use. Also, there is a limitation on the size of the arguments passed to a global kernel, so if you have a very large class this is simply impossible. So we make another attempt.

.. code-block:: cpp

   __global__ void callVirtualFunction(MyBaseClass* myBaseClass) {
      myBaseClass->getValue();
   }

   int main(int argc, char** argv) {
      MyBaseClass* myBaseClass = new MyDerivedClass(0);
      MyBaseClass* d_myBaseClass;
      cudaMalloc(&d_myBaseClass, sizeof(MyBaseClass));
      cudaMemcpy(d_myBaseClass, myBaseClass, sizeof(MyBaseClass), cudaMemcpyHostToDevice);

      callVirtualFunction<<<1, 1>>>(d_myBaseClass);

      cudaFree(d_myBaseClass);
      delete myBaseClass;

      return 0;
   }

We are getting nearer, but there is still a flaw. The bits of `myBaseClass` contain the virtual function table that allows virtual function lookups on the host, but that virtual function table is not valid for lookups on the device since it contains pointers to host functions. It will not work any better to cast to `MyDerivedClass` and copy the bits. The only option is to call the constructor on the device and keep that device pointer around.

.. code-block:: cpp

   __global__ void make_on_device(MyBaseClass** myBaseClass, int argument) {
      *myBaseClass = new MyDerivedClass(argument);
   }

   __global__ void destroy_on_device(MyBaseClass* myBaseClass) {
      delete myBaseClass;
   }

   __global__ void callVirtualFunction(MyBaseClass* myBaseClass) {
      myBaseClass->getValue();
   }

   int main(int argc, char** argv) {
      MyBaseClass** d_temp;
      cudaMalloc(&d_temp, sizeof(MyBaseClass*));
      make_on_device<<<1, 1>>>(d_temp, 0);

      MyBaseClass** temp = (MyBaseClass**) malloc(sizeof(MyBaseClass*));
      cudaMemcpy(temp, d_temp, sizeof(MyBaseClass*), cudaMemcpyDeviceToHost);
      MyBaseClass d_myBaseClass = *temp;

      callVirtualFunction<<<1, 1>>>(d_myBaseClass);

      free(temp);
      destroy_on_device<<<1, 1>>>(d_myBaseClass);
      cudaFree(d_temp);

      return 0;
   }

OK, this is finally correct, but super tedious. So we took care of all the boilerplate and underlying details for you. The final result is at least recognizable when compared to the original code. The added benefit is that you can use a `chai::managed_ptr` on the host AND the device.

.. code-block:: cpp

   __global__ void callVirtualFunction(chai::managed_ptr<MyBaseClass> myBaseClass) {
      myBaseClass->getValue();
   }

   int main(int argc, char** argv) {
      chai::managed_ptr<MyBaseClass> myBaseClass = chai::make_managed<MyDerivedClass>(0);
      myBaseClass->getValue(); // Accessible on the host
      callVirtualFunction<<<1, 1>>>(myBaseClass); // Accessible on the device
      myBaseClass.free();
      return 0;
   }

OK, so we didn't do all the work for you, but we definitely gave you a leg up. What's left for you to do? You just need to make sure the functions accessed on the device have the `__device__` specifier (including constructors and destructors). We use the `CHAI_HOST_DEVICE` macro in this example, which actually annotates the functions as `__host__ __device__` so we can call the virtual method on both the host and the device. You also need to make sure the destructors of all base classes are virtual so the object gets cleaned up properly on the device.

.. code-block:: cpp

   class MyBaseClass {
      public:
         CARE_HOST_DEVICE MyBaseClass() {}
         CARE_HOST_DEVICE virtual ~MyBaseClass() {}
         CARE_HOST_DEVICE virtual int getValue() const = 0;
   };

   class MyDerivedClass : public MyBaseClass {
      public:
         CARE_HOST_DEVICE MyDerivedClass(int value) : MyBaseClass(), m_value(value) {}
         CARE_HOST_DEVICE ~MyDerivedClass() {}
         CARE_HOST_DEVICE int getValue() const { return m_value; }

      private:
         int m_value;
   };

Now you may rightfully ask, what happens when this class contains raw pointers? There is a convenient solution for this case and we demonstrate with a more interesting example.

.. code-block:: cpp

   class MyBaseClass {
      public:
         CARE_HOST_DEVICE MyBaseClass() {}
         CARE_HOST_DEVICE virtual ~MyBaseClass() {}
         CARE_HOST_DEVICE virtual int getScalarValue() const = 0;
         CARE_HOST_DEVICE virtual int getArrayValue(int index) const = 0;
   };

   class MyDerivedClass : public MyBaseClass {
      public:
         CARE_HOST_DEVICE MyDerivedClass(int scalarValue, int* arrayValue)
            : MyBaseClass(), m_scalarValue(scalarValue), m_arrayValue(arrayValue) {}
         CARE_HOST_DEVICE ~MyDerivedClass() {}
         CARE_HOST_DEVICE int getScalarValue() const { return m_scalarValue; }
         CARE_HOST_DEVICE int getArrayValue() const { return m_arrayValue; }

      private:
         int m_scalarValue;
         int* m_arrayValue;
   };

   __global__ void callVirtualFunction(chai::managed_ptr<MyBaseClass> myBaseClass) {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      myBaseClass->getScalarValue();
      myBaseClass->getArrayValue(i);
   }

   int main(int argc, char** argv) {
      chai::ManagedArray<int> arrayValue(10);
      chai::managed_ptr<MyBaseClass> myBaseClass
         = chai::make_managed<MyDerivedClass>(0, chai::unpack(arrayValue));
      callVirtualFunction<<<1, 10>>>(myBaseClass);
      myBaseClass.free();
      arrayValue.free();
      return 0;
   }

The respective host and device pointers contained in the `chai::ManagedArray` can be extracted and passed to the host and device instance of `MyDerivedClass` using `chai::unpack`. Of course, if you never dereference `m_arrayValue` on the device, you could simply pass a raw pointer to `chai::make_managed`. If the class contains a `chai::ManagedArray`, a `chai::ManagedArray` can simply be passed to the constructor. The same rules apply for passing a `chai::managed_ptr`, calling `chai::unpack` on a `chai::managed_ptr`, or passing a raw pointer and not accessing it on the device.

More complicated rules apply for keeping the data in sync between the host and device instances of an object, but it is possible to do so to a limited extent. It is also possible to control the lifetimes of objects passed to `chai::make_managed`.

.. code-block:: cpp
   int main(int argc, char** argv) {
      chai::ManagedArray<int> arrayValue(10);

      chai::managed_ptr<MyBaseClass> myBaseClass
         = chai::make_managed<MyDerivedClass>(0, chai::unpack(arrayValue));
      myBaseClass.set_callback([=] (chai::Action action, chai::ExecutionSpace space, void*) mutable {
         if (action == chai::ACTION_MOVE) {
            (void) chai::ManagedArray<int> temp(arrayValue); // Copy constructor triggers movement
         }
         else if (action == chai::ACTION_FREE && space == chai::NONE) {
            temp.free();
         }

         return false;
      });

      callVirtualFunction<<<1, 10>>>(myBaseClass);
      myBaseClass.free();
      // arrayValue.free(); // Not needed anymore
      return 0;
   }
