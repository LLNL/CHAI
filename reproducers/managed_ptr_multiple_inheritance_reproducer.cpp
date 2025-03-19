//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
//
// To reproduce crash on with a ROCM compiler:
// % mkdir build_rocm
// % cd build_rocm
// % cmake -DCHAI_ENABLE_REPRODUCERS=1 -C ../host-configs/lc/toss_4_x86_64_ib_cray/amdclang.cmake ..
// % flux alloc -N 1 -n 1 -g1 make -j 40
// % flux alloc -N 1 -n 1 -g1 ./bin/managed_ptr_multiple_inheritance_reproducer.exe
//
// - Note that the "this" pointer in YofXfromRTTable1D::RootFromBaseX differs from the
//   "this" pointer in other YofXfromRTTable1D methods. The crash occurs because m_table is null
//   and is dereferenced.
// - Interestingly , the crash goes away if GetNumStrings() is removed
//
// The NVCC case does not crash and has consistent pointer addresses in YofXfromRTTable1D.
// This can be reproduced with:
// % mkdir build_cuda
// % cd build_cuda
// % cmake -DCHAI_ENABLE_REPRODUCERS=1 -C ../host-configs/lc/blueos_3_ppc64le_ib_p9/nvcc_clang.cmake ..
// % lalloc 1 make -j 40
// % lalloc 1 ./bin/managed_ptr_multiple_inheritance_reproducer.exe
//

#include "chai/config.hpp"
#include "chai/managed_ptr.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

class  Table {
   public:
      CHAI_HOST_DEVICE Table() {
#ifdef CHAI_DEVICE_COMPILE
         printf("Table::Table POINTER %p\n", this) ;
#endif
      }

      CHAI_HOST_DEVICE virtual ~Table() {}

      class  Data {
         public:
            CHAI_HOST_DEVICE Data() {
#ifdef CHAI_DEVICE_COMPILE
               printf("Table::Data::Data POINTER %p\n", this) ;
#endif
            }

            CHAI_HOST_DEVICE virtual ~Data() {}

         private:
      } ;

      class  Derived {
         public:
            CHAI_HOST_DEVICE Derived() {
#ifdef CHAI_DEVICE_COMPILE
               printf("Table::Derived::Derived POINTER %p\n", this) ;
#endif
            }

            CHAI_HOST_DEVICE virtual ~Derived() {}

            /// Removing this makes the bug go away!!!
            virtual int GetNumStrings() const = 0 ;
      } ;

      class  Compute {
         public:
            CHAI_HOST_DEVICE Compute() {
#ifdef CHAI_DEVICE_COMPILE
               printf("Table::Compute::Compute POINTER %p\n", this) ;
#endif
            }

            CHAI_HOST_DEVICE virtual ~Compute() {}

         protected:

            CHAI_HOST_DEVICE double BaseXFromNewX() const {
#ifdef CHAI_DEVICE_COMPILE
               printf("Table::Compute::BaseXFromNewX %p\n", this) ;
#endif
               RootFromBaseX() ;
               return 0.0 ;
            }

            CHAI_HOST_DEVICE virtual double RootFromBaseX() const = 0;

         private:
            Compute(const Compute &other);
            Compute &operator=(const Compute &other);
      } ;
   private:

      Table(const Table &other);

      Table& operator =(const Table&);
};

class  SimpleTable : public Table {
   public:
      CHAI_HOST_DEVICE SimpleTable() : Table() {
#ifdef CHAI_DEVICE_COMPILE
         printf("SimpleTable::SimpleTable POINTER %p\n", this) ;
#endif
      }


      CHAI_HOST_DEVICE virtual ~SimpleTable() {}

   private:
      SimpleTable(const SimpleTable& other);

      SimpleTable& operator =(const SimpleTable&);
};

class  Table1D : public SimpleTable {
   public:
      CHAI_HOST_DEVICE Table1D() : SimpleTable() {
#ifdef CHAI_DEVICE_COMPILE
         printf("Table1D::Table1D POINTER %p\n", this) ;
#endif
      }

      CHAI_HOST_DEVICE virtual ~Table1D() {}

      CHAI_HOST_DEVICE virtual double Evaluate() const = 0 ;

   protected:
      CHAI_HOST_DEVICE Table1D(const Table1D &other) = delete;

      CHAI_HOST_DEVICE Table1D& operator=(const Table1D &other) = delete;
} ;

class  DataTable1D : public Table1D, public Table::Data {
   public:
      CHAI_HOST_DEVICE DataTable1D() : Table1D(), Table::Data() {}

      CHAI_HOST_DEVICE virtual ~DataTable1D() {}

      CHAI_HOST_DEVICE virtual double Evaluate() const override {
         return innerEvaluate() ;
      }

   protected:
      CHAI_HOST_DEVICE virtual double innerEvaluate() const = 0 ;
} ;

class LinearTable1D : public DataTable1D {
   public:
      CHAI_HOST_DEVICE  LinearTable1D() : DataTable1D() {}

      CHAI_HOST_DEVICE virtual ~LinearTable1D() {}

      CHAI_HOST_DEVICE virtual double innerEvaluate() const override {
         return 0.0 ;
      }

   private:
      CHAI_HOST_DEVICE LinearTable1D(const LinearTable1D &other) = delete;
      CHAI_HOST_DEVICE LinearTable1D &operator=(const LinearTable1D &other) = delete;
} ;


class  DerivedTable1D : public Table1D, public Table::Data, public Table::Derived {
   public:
      CHAI_HOST_DEVICE DerivedTable1D() : Table1D(), Table::Data(), Table::Derived() {
#ifdef CHAI_DEVICE_COMPILE
         printf("DerivedTable1D::DerivedTable1D POINTER %p\n", this) ;
#endif
      }

      CHAI_HOST_DEVICE virtual ~DerivedTable1D() {}

      CHAI_HOST_DEVICE virtual double Evaluate() const override {
#ifdef CHAI_DEVICE_COMPILE
         printf("DerivedTable1D::Evaluate POINTER %p\n", this) ;
#endif
         return innerEvaluate() ;
      }

      virtual int GetNumStrings() const override = 0;

   protected:

      CHAI_HOST_DEVICE virtual double innerEvaluate() const = 0 ;

   private:
      CHAI_HOST_DEVICE DerivedTable1D(const DerivedTable1D &other) = delete;
      CHAI_HOST_DEVICE DerivedTable1D &operator=(const DerivedTable1D &other) = delete;
} ;

class StitchTable1D : public DerivedTable1D {
   public:
      CHAI_HOST_DEVICE  StitchTable1D(int nt, chai::managed_ptr<Table1D const>* tabs) : DerivedTable1D(), m_nTables(nt), m_tables(new const Table1D *[nt]) {
         for (int i = 0 ; i < nt ; ++i) {
            m_tables[i] = tabs[i].get() ;
#ifdef CHAI_DEVICE_COMPILE
            printf("StitchTable1D::StitchTable1D POINTER %p i %d m_tables[i] %p\n", this, i, m_tables[i]) ;
#endif
         }
      }

      CHAI_HOST_DEVICE virtual ~StitchTable1D() {
         if (m_nTables > 0) {
            delete [] m_tables ;
         }
      }

      inline virtual int GetNumStrings() const override { return m_nTables; }

   private:
      CHAI_HOST_DEVICE double innerEvaluate() const override {
         for (int i = 0 ; i < m_nTables ; ++i) {
#ifdef CHAI_DEVICE_COMPILE
            printf("StitchTable1D::innerEvaluate POINTER %p i %d m_tables[i] %p\n", this, i, m_tables[i]) ;
#endif
            m_tables[i]->Evaluate() ;
         }
         return 0.0 ;
      }

      CHAI_HOST_DEVICE StitchTable1D() = delete;
      CHAI_HOST_DEVICE StitchTable1D(const StitchTable1D &other) = delete;
      CHAI_HOST_DEVICE StitchTable1D &operator=(const StitchTable1D &other) = delete;

      int m_nTables ;
      const Table1D ** m_tables = nullptr;
} ;

class ComputedTable1D : public DerivedTable1D, public Table::Compute {
   public:
      CHAI_HOST_DEVICE  ComputedTable1D(const Table1D * f) : DerivedTable1D(), Table::Compute(), m_table(f)
      {
#ifdef CHAI_DEVICE_COMPILE
         printf("ComputedTable1D::ComputedTable1D POINTER %p m_table %p\n", this, m_table) ;
#endif
      }

      CHAI_HOST_DEVICE virtual ~ComputedTable1D() {}

   protected:
      const Table1D *m_table;

   private:
      CHAI_HOST_DEVICE ComputedTable1D() = delete;
      CHAI_HOST_DEVICE ComputedTable1D(const ComputedTable1D &other) = delete;
      CHAI_HOST_DEVICE ComputedTable1D& operator=(const ComputedTable1D &other) = delete;
} ;

class YofXfromRTTable1D : public ComputedTable1D {
   public:
      CHAI_HOST_DEVICE YofXfromRTTable1D(Table1D const * f) : ComputedTable1D(f) {
#ifdef CHAI_DEVICE_COMPILE
         printf("YofXfromRTTable1D::YofXfromRTTable1D POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
         m_table->Evaluate() ;
      }

      CHAI_HOST_DEVICE virtual ~YofXfromRTTable1D() {}

      inline int virtual GetNumStrings() const override { return 1; }

   private:
      CHAI_HOST_DEVICE virtual double innerEvaluate() const override {
#ifdef CHAI_DEVICE_COMPILE
         printf("YofXfromRTTable1D::innerEvaluate POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
         BaseXFromNewX() ;
         return 0.0 ;
      }

      CHAI_HOST_DEVICE virtual double RootFromBaseX() const override {
#ifdef CHAI_DEVICE_COMPILE
         printf("YofXfromRTTable1D::RootFromBaseX POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
         return m_table->Evaluate() ;
      }

      CHAI_HOST_DEVICE YofXfromRTTable1D() = delete;
      CHAI_HOST_DEVICE YofXfromRTTable1D(const YofXfromRTTable1D &other) = delete;
      CHAI_HOST_DEVICE YofXfromRTTable1D &operator=(const YofXfromRTTable1D &other) = delete;
} ;

class RofTfromXYTable1D : public ComputedTable1D {
   public:
      CHAI_HOST_DEVICE RofTfromXYTable1D(Table1D const * f) : ComputedTable1D(f) {
#ifdef CHAI_DEVICE_COMPILE
         printf("RofTfromXY::RofTfromXYTable1D POINTER %p m_table %p\n", this, m_table) ;
#endif

         m_table->Evaluate() ;
      }

      CHAI_HOST_DEVICE virtual ~RofTfromXYTable1D() {}

      inline int virtual GetNumStrings() const override { return 1; }

   private:
      CHAI_HOST_DEVICE virtual double innerEvaluate() const override {
         BaseXFromNewX() ;
         return 0.0 ;
      }

      CHAI_HOST_DEVICE virtual double RootFromBaseX() const override {
         return m_table->Evaluate() ;
      }

      CHAI_HOST_DEVICE RofTfromXYTable1D() = delete;
      CHAI_HOST_DEVICE RofTfromXYTable1D(const RofTfromXYTable1D &other) = delete;
      CHAI_HOST_DEVICE RofTfromXYTable1D &operator=(const RofTfromXYTable1D &other) = delete;
} ;

int main(int, char**) {
   chai::managed_ptr<Table1D const> tabArray[6] ;

   tabArray[0] = chai::make_managed<LinearTable1D>() ;
   tabArray[1] = chai::make_managed<YofXfromRTTable1D>(chai::unpack(tabArray[0])) ;
   tabArray[2] = chai::make_managed<LinearTable1D>() ;
   tabArray[3] = chai::make_managed<LinearTable1D>() ;

   chai::ManagedArray<chai::managed_ptr<const Table1D> > host_device_temp0(2) ;
   chai::managed_ptr<const Table1D> *host_temp0 = host_device_temp0.data() ;
   host_temp0[0] = tabArray[1] ;
   host_temp0[1] = tabArray[2] ;

   tabArray[4] = chai::make_managed<StitchTable1D>(2, chai::unpack(host_device_temp0)) ;

   chai::ManagedArray<chai::managed_ptr<const Table1D> > host_device_temp1(2) ;
   chai::managed_ptr<const Table1D> *host_temp1 = host_device_temp1.data() ;
   host_temp1[0] = tabArray[3] ;
   host_temp1[1] = tabArray[4] ;

   tabArray[5] = chai::make_managed<StitchTable1D>(2, chai::unpack(host_device_temp1)) ;

   chai::make_managed<RofTfromXYTable1D>(chai::unpack(tabArray[5])) ;

   printf("SUCCESS\n") ;

   host_device_temp0.free() ;
   host_device_temp1.free() ;

   return 0;
}

