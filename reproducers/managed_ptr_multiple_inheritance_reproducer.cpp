//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/config.hpp"
#include "chai/managed_ptr.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

class DataTable1D;
class DerivedTable1D;
typedef double real8 ;

class Table ;
class SimpleTable;
class Table1D ;

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

            CHAI_HOST_DEVICE real8 BaseXFromNewX() const ;

            CHAI_HOST_DEVICE virtual real8 RootFromBaseX() const = 0;

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

      CHAI_HOST_DEVICE virtual real8 Evaluate() const = 0 ;

   protected:
      CHAI_HOST_DEVICE Table1D(const Table1D &other) = delete;

      CHAI_HOST_DEVICE Table1D& operator=(const Table1D &other) = delete;
} ;

class  DataTable1D : public Table1D, public Table::Data {
   public:
      CHAI_HOST_DEVICE DataTable1D() : Table1D(), Table::Data() {}

      CHAI_HOST_DEVICE virtual ~DataTable1D() {}

      CHAI_HOST_DEVICE virtual real8 Evaluate() const override;

      using Table1D::Evaluate ;

   protected:
      CHAI_HOST_DEVICE virtual real8 innerEvaluate() const = 0 ;
} ;

class LinearTable1D : public DataTable1D {
   public:
      CHAI_HOST_DEVICE  LinearTable1D() : DataTable1D() {}

      CHAI_HOST_DEVICE virtual ~LinearTable1D() {}

      CHAI_HOST_DEVICE virtual real8 innerEvaluate() const override;

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

      CHAI_HOST_DEVICE virtual real8 Evaluate() const override;

      virtual int GetNumStrings() const override = 0;

   protected:

      CHAI_HOST_DEVICE virtual real8 innerEvaluate() const = 0 ;

   private:
      CHAI_HOST_DEVICE DerivedTable1D(const DerivedTable1D &other) = delete;
      CHAI_HOST_DEVICE DerivedTable1D &operator=(const DerivedTable1D &other) = delete;
} ;

class StitchTable1D : public DerivedTable1D {
   public:
      CHAI_HOST_DEVICE  StitchTable1D(int nt, chai::managed_ptr<Table1D const>* tabs) ;

      CHAI_HOST_DEVICE virtual ~StitchTable1D() ;

      inline virtual int GetNumStrings() const override { return m_nTables; }

   private:
      CHAI_HOST_DEVICE real8 innerEvaluate() const override;

      CHAI_HOST_DEVICE StitchTable1D() = delete;
      CHAI_HOST_DEVICE StitchTable1D(const StitchTable1D &other) = delete;
      CHAI_HOST_DEVICE StitchTable1D &operator=(const StitchTable1D &other) = delete;

      int m_nTables ;
      const Table1D ** m_tables = nullptr;
} ;

class ComputedTable1D : public DerivedTable1D, public Table::Compute {
   public:
      CHAI_HOST_DEVICE  ComputedTable1D(const Table1D * table);

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
      CHAI_HOST_DEVICE YofXfromRTTable1D(Table1D const * f) ;

      CHAI_HOST_DEVICE virtual ~YofXfromRTTable1D() {}

      inline int virtual GetNumStrings() const override { return 1; }

   private:
      CHAI_HOST_DEVICE virtual real8 innerEvaluate() const override;

      CHAI_HOST_DEVICE virtual real8 RootFromBaseX() const override;

      CHAI_HOST_DEVICE YofXfromRTTable1D() = delete;
      CHAI_HOST_DEVICE YofXfromRTTable1D(const YofXfromRTTable1D &other) = delete;
      CHAI_HOST_DEVICE YofXfromRTTable1D &operator=(const YofXfromRTTable1D &other) = delete;
} ;

class RofTfromXYTable1D : public ComputedTable1D {
   public:
      CHAI_HOST_DEVICE RofTfromXYTable1D(Table1D const * f) ;

      CHAI_HOST_DEVICE virtual ~RofTfromXYTable1D() {}

      inline int virtual GetNumStrings() const override { return 1; }

   private:
      CHAI_HOST_DEVICE virtual real8 innerEvaluate() const override;

      CHAI_HOST_DEVICE virtual real8 RootFromBaseX() const override;

      CHAI_HOST_DEVICE RofTfromXYTable1D() = delete;
      CHAI_HOST_DEVICE RofTfromXYTable1D(const RofTfromXYTable1D &other) = delete;
      CHAI_HOST_DEVICE RofTfromXYTable1D &operator=(const RofTfromXYTable1D &other) = delete;
} ;

CHAI_HOST_DEVICE real8 Table::Compute::BaseXFromNewX() const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("Table::Compute::BaseXFromNewX %p\n", this) ;
#endif
   RootFromBaseX() ;
   return 0.0 ;
}

CHAI_HOST_DEVICE real8 LinearTable1D::innerEvaluate() const
{
   return 0.0 ;
}

CHAI_HOST_DEVICE real8 DerivedTable1D::Evaluate() const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("DerivedTable1D::Evaluate POINTER %p\n", this) ;
#endif
   return innerEvaluate() ;
}

CHAI_HOST_DEVICE real8 DataTable1D::Evaluate() const
{
   return innerEvaluate() ;
}

CHAI_HOST_DEVICE StitchTable1D::StitchTable1D(int nt, chai::managed_ptr<Table1D const>* tabs)
   : DerivedTable1D()
   , m_nTables(nt)
   , m_tables(new const Table1D *[nt])
{
   for (int i = 0 ; i < nt ; ++i) {
      m_tables[i] = tabs[i].get() ;
#ifdef CHAI_DEVICE_COMPILE
      printf("StitchTable1D::StitchTable1D POINTER %p i %d m_tables[i] %p\n", this, i, m_tables[i]) ;
#endif
   }
}

CHAI_HOST_DEVICE StitchTable1D::~StitchTable1D()
{
   if (m_nTables > 0) {
      delete [] m_tables ;
   }
}

CHAI_HOST_DEVICE real8 StitchTable1D::innerEvaluate() const
{
   for (int i = 0 ; i < m_nTables ; ++i) {
#ifdef CHAI_DEVICE_COMPILE
      printf("StitchTable1D::innerEvaluate POINTER %p i %d m_tables[i] %p\n", this, i, m_tables[i]) ;
#endif
      m_tables[i]->Evaluate() ;
   }
   return 0.0 ;
}

CHAI_HOST_DEVICE ComputedTable1D::
ComputedTable1D(const Table1D * f)
   : DerivedTable1D(), Table::Compute(),
   m_table(f)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("ComputedTable1D::ComputedTable1D POINTER %p m_table %p\n", this, m_table) ;
#endif
}

CHAI_HOST_DEVICE YofXfromRTTable1D::YofXfromRTTable1D(Table1D const * f)
   : ComputedTable1D(f)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("YofXfromRTTable1D::YofXfromRTTable1D POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
   m_table->Evaluate() ;
}

CHAI_HOST_DEVICE real8 YofXfromRTTable1D::RootFromBaseX() const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("YofXfromRTTable1D::RootFromBaseX POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
   return m_table->Evaluate() ;
}

CHAI_HOST_DEVICE real8 YofXfromRTTable1D::innerEvaluate() const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("YofXfromRTTable1D::innerEvaluate POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
   BaseXFromNewX() ;
   return 0.0 ;
}

CHAI_HOST_DEVICE RofTfromXYTable1D::RofTfromXYTable1D(Table1D const * f)
   : ComputedTable1D(f)

{
#ifdef CHAI_DEVICE_COMPILE
   printf("RofTfromXY::RofTfromXYTable1D POINTER %p m_table %p\n", this, m_table) ;
#endif

   m_table->Evaluate() ;
}


CHAI_HOST_DEVICE real8 RofTfromXYTable1D::RootFromBaseX() const
{
   return m_table->Evaluate() ;
}

CHAI_HOST_DEVICE real8 RofTfromXYTable1D::innerEvaluate() const
{
   BaseXFromNewX() ;
   return 0.0 ;
}

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

