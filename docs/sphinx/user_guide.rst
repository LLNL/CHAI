.. Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
 rights reserved.
 
 Produced at the Lawrence Livermore National Laboratory
 
 This file is part of CHAI.
 
 LLNL-CODE-705877
 
 For details, see https:://github.com/LLNL/CHAI
 Please also see the NOTICE and LICENSE files.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 
 - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.
 
 - Neither the name of the LLNS/LLNL nor the names of its contributors
   may be used to endorse or promote products derived from this
   software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

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
         MyDerivedClass(const int value) : MyBaseClass(), m_value(value) {}
         virtual ~MyDerivedClass() {}
         virtual int getValue() const { return m_value; }

      private:
         int m_value;
   };

   int main(int argc, char** argv) {
      MyBaseClass* myBaseClass = new MyDerivedClass(0);
      myBaseClass->getValue();
      return 0;
   }

It is perfectly fine to call `myBaseClass->getValue()` in host code, since myBaseClass was created on the host. However, what if you want to call this virtual function on the device?

.. code-block:: cpp

   __global__ void callVirtualFunction(MyBaseClass* myBaseClass) {
      myBaseClass->getValue();
   }

   int main(int argc, char** argv) {
      MyBaseClass* myBaseClass = new MyDerivedClass(0);
      callVirtualFunction<<<1, 1>>>(myBaseClass);
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
      return 0;
   }

At first glance, this may seem like it would work, but there is a flaw - copy constructors are not virtual. You could cast to MyDerivedClass and then pass that by value, but if there are tons of classes in this heirarchy, how do you know which one to cast it to? You could try dynamic_cast dozens of times, but that is not performant or sustainable. You could also write a virtual clone method, but that is also not sustainable. You could refactor to use the curiously recurring template pattern, but that would likely require a large development effort. Also, there is a limitation on the size of the arguments passed to a global kernel, so if you have a very large class, this is simply impossible. So we make another attempt.

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
      return 0;
   }

We are getting nearer, but there is still a flaw. The bits of myBaseClass contain the virtual function table that allows virtual function lookups on the host, but that virtual function table is not valid for lookups on the device since it contains pointers to host functions. It will not work any better to cast to MyDerivedClass and copy the bits. The only option is to call the constructor on the device and keep that device pointer around.

.. code-block:: cpp

   __global__ void make_on_device(MyBaseClass** myBaseClass, int argument) {
      *myBaseClass = new MyDerivedClass(argument);
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
      cudaFree(d_temp);

      // Still need to call delete on the device
      return 0;
   }

OK, this is finally correct, but super tedious. So we took care of the details for you.

.. code-block:: cpp

   __global__ void callVirtualFunction(chai::managed_ptr<MyBaseClass> myBaseClass) {
      myBaseClass->getValue();
   }

   int main(int argc, char** argv) {
      chai::managed_ptr<MyBaseClass> myBaseClass = chai::make_managed<MyDerivedClass>(0);
      callVirtualFunction<<<1, 1>>>(myBaseClass);
      myBaseClass.free();

      return 0;
   }

OK, so we didn't do all the work for you, but we definitely gave you a leg up. What's left for you to do? You just need to make sure the functions accessed on the device have the __device__ specifier (including constructors and destructors). You also need to make sure the destructors are virtual so the object gets cleaned up properly on the device.

.. code-block:: cpp

   class MyBaseClass {
      public:
         CARE_HOST_DEVICE MyBaseClass() {}
         CARE_HOST_DEVICE virtual ~MyBaseClass() {}
         CARE_HOST_DEVICE virtual int getValue() const = 0;
   };

   class MyDerivedClass : public MyBaseClass {
      public:
         CARE_HOST_DEVICE MyDerivedClass(const int value) : MyBaseClass(), m_value(value) {}
         CARE_HOST_DEVICE virtual ~MyDerivedClass() {}
         CARE_HOST_DEVICE virtual int getValue() const { return m_value; }

      private:
         int m_value;
   };
