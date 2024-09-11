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
      CHAI_HOST_DEVICE Table(const char *id) ;

      CHAI_HOST_DEVICE virtual ~Table() ;

      class  Data {
         public:
            CHAI_HOST_DEVICE Data() {
#ifdef CHAI_DEVICE_COMPILE
               printf("Table::Data::Data %p\n", this) ;
#endif
            }

            CHAI_HOST_DEVICE virtual ~Data() {}

         private:
      } ;

      class  Derived {
         public:
            CHAI_HOST_DEVICE Derived() {
#ifdef CHAI_DEVICE_COMPILE
               printf("Table::Derived::Derived %p\n", this) ;
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
               printf("Table::Compute::Compute %p\n", this) ;
#endif
            }

            CHAI_HOST_DEVICE virtual ~Compute() {}

         protected:

            CHAI_HOST_DEVICE real8 BaseXFromNewX(const real8 *newX) const ;

            CHAI_HOST_DEVICE virtual real8 RootFromBaseX(real8 baseX, const real8 *vals) const = 0;

         private:
            Compute(const Compute &other);
            Compute &operator=(const Compute &other);
      } ;
   protected:

      CHAI_HOST_DEVICE int Search(real8 xin, const real8* xvals, int numX) const ;

   private:

      const char *m_id ;

      Table() ;

      Table(const Table &other);

      Table& operator =(const Table&);
};

class  SimpleTable : public Table {
   public:
      CHAI_HOST_DEVICE SimpleTable(const char* id);

      CHAI_HOST_DEVICE virtual ~SimpleTable() {}

   private:
      SimpleTable();

      SimpleTable(const SimpleTable& other);

      SimpleTable& operator =(const SimpleTable&);
};

class  Table1D : public SimpleTable {
   public:
      CHAI_HOST_DEVICE Table1D(const char* id) ;

      CHAI_HOST_DEVICE virtual ~Table1D() {}

      CHAI_HOST_DEVICE virtual real8 Evaluate(real8 xin) const = 0 ;

   protected:

      CHAI_HOST_DEVICE Table1D() = delete;

      CHAI_HOST_DEVICE Table1D(const Table1D &other) = delete;

      CHAI_HOST_DEVICE Table1D& operator=(const Table1D &other) = delete;
} ;

class  DataTable1D : public Table1D, public Table::Data {
   public:
      CHAI_HOST_DEVICE DataTable1D(const int numX,
                                  const char *id,
                                  const real8 *x,
                                  const real8 *fx) ;

      CHAI_HOST_DEVICE virtual ~DataTable1D();

      CHAI_HOST_DEVICE real8 Evaluate(real8 xin) const override;

      using Table1D::Evaluate ;

      CHAI_HOST_DEVICE real8 GetXAt(int pos) const;

      CHAI_HOST_DEVICE real8 GetYAt(int pos) const;

   protected:
      CHAI_HOST_DEVICE virtual real8 innerEvaluate(real8 x) const = 0 ;

      CHAI_HOST_DEVICE int SearchX(real8 xin) const ;

      const int m_numX ;

      real8 *m_x ;

      real8 *m_fx ;
} ;

class LinearTable1D : public DataTable1D {
   public:
      CHAI_HOST_DEVICE  LinearTable1D(const int numX,
                                                   const char *id,
                                                   const real8 *x,
                                                   const real8 *fx) ;

      CHAI_HOST_DEVICE virtual ~LinearTable1D() ;

      CHAI_HOST_DEVICE real8 innerEvaluate(real8 x) const override;

   private:
      CHAI_HOST_DEVICE LinearTable1D() = delete;
      CHAI_HOST_DEVICE LinearTable1D(const LinearTable1D &other) = delete;
      CHAI_HOST_DEVICE LinearTable1D &operator=(const LinearTable1D &other) = delete;

      real8 *m_x0 ;
      real8 *m_xd ;
      real8 *m_fx0 ;
      real8 *m_fxd ;
} ;


class  DerivedTable1D : public Table1D, public Table::Data, public Table::Derived {
   public:
      CHAI_HOST_DEVICE DerivedTable1D(const char *id) ;

      CHAI_HOST_DEVICE virtual ~DerivedTable1D() {}

      CHAI_HOST_DEVICE real8 Evaluate(real8 xin) const override;

      virtual int GetNumStrings() const override = 0;

   protected:

      CHAI_HOST_DEVICE virtual real8 innerEvaluate(real8 x) const = 0 ;

   private:
      CHAI_HOST_DEVICE DerivedTable1D() = delete;
      CHAI_HOST_DEVICE DerivedTable1D(const DerivedTable1D &other) = delete;
      CHAI_HOST_DEVICE DerivedTable1D &operator=(const DerivedTable1D &other) = delete;
} ;

class StitchTable1D : public DerivedTable1D {
   public:
      CHAI_HOST  StitchTable1D(const char* id,
                                               int nt,
                                               Table1D const** tabs,
                                               const real8* bds) ;

      CHAI_HOST_DEVICE  StitchTable1D(const char* id,
                                                   int nt,
                                                   chai::managed_ptr<Table1D const>* tabs,
                                                   const real8* bds) ;

      CHAI_HOST_DEVICE virtual ~StitchTable1D() ;

      inline int GetNumStrings() const override { return m_nTables; }

   private:
      CHAI_HOST_DEVICE real8 innerEvaluate(real8 x) const override;

      CHAI_HOST_DEVICE StitchTable1D() = delete;
      CHAI_HOST_DEVICE StitchTable1D(const StitchTable1D &other) = delete;
      CHAI_HOST_DEVICE StitchTable1D &operator=(const StitchTable1D &other) = delete;

      int m_nTables ;
      const Table1D ** m_tables = nullptr;
      real8 * m_bds ;
} ;

class ComputedTable1D : public DerivedTable1D, public Table::Compute {
   public:
      CHAI_HOST_DEVICE  ComputedTable1D(const char* id, const Table1D * table,
                                                     real8 lb, real8 ub);

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
      CHAI_HOST_DEVICE YofXfromRTTable1D(const char* id, Table1D const * f,
                                        real8 theta_low, real8 theta_high) ;

      CHAI_HOST_DEVICE virtual ~YofXfromRTTable1D() {}

      inline int GetNumStrings() const override { return 1; }

   private:
      CHAI_HOST_DEVICE real8 innerEvaluate(real8 x) const override;

      CHAI_HOST_DEVICE real8 RootFromBaseX(real8 baseX, const real8 *vals) const override;

      CHAI_HOST_DEVICE YofXfromRTTable1D() = delete;
      CHAI_HOST_DEVICE YofXfromRTTable1D(const YofXfromRTTable1D &other) = delete;
      CHAI_HOST_DEVICE YofXfromRTTable1D &operator=(const YofXfromRTTable1D &other) = delete;
} ;

class RofTfromXYTable1D : public ComputedTable1D {
   public:
      CHAI_HOST_DEVICE RofTfromXYTable1D(const char* id, Table1D const * f,
                                        real8 x_low, real8 x_high) ;

      CHAI_HOST_DEVICE virtual ~RofTfromXYTable1D() {}

      inline int GetNumStrings() const override { return 1; }

   private:
      CHAI_HOST_DEVICE real8 innerEvaluate(real8 x) const override;

      CHAI_HOST_DEVICE real8 RootFromBaseX(real8 baseX, const real8 *vals) const override;

      CHAI_HOST_DEVICE RofTfromXYTable1D() = delete;
      CHAI_HOST_DEVICE RofTfromXYTable1D(const RofTfromXYTable1D &other) = delete;
      CHAI_HOST_DEVICE RofTfromXYTable1D &operator=(const RofTfromXYTable1D &other) = delete;
} ;

CHAI_HOST_DEVICE Table::Table(const char *id)
   : 
   m_id(id)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("Table::Table %p\n", this) ;
#endif
}

CHAI_HOST_DEVICE Table::~Table()
{
#ifndef  CHAI_DEVICE_COMPILE
   if (m_id != nullptr) {
      delete[] m_id ;
      m_id = nullptr ;
   }
#endif
}

CHAI_HOST_DEVICE real8 Table::Compute::BaseXFromNewX(const real8 *xin) const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("Table::Compute::BaseXFromNewX %p\n", this) ;
#endif
   RootFromBaseX(0.0, xin) ;
   return 0.0 ;
}

CHAI_HOST_DEVICE int Table::Search(real8 xin, const real8* xvals, int numX) const
{
   int i = 0 ;
   while (i < numX - 2) {
      if (xin <= xvals[i+1]) {
         break ;
      }
      i++ ;
   }
   return i ;
}

CHAI_HOST_DEVICE SimpleTable::SimpleTable(const char* id)
   : Table(id)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("SimpleTable::SimpleTable %p\n", this) ;
#endif
}

CHAI_HOST_DEVICE LinearTable1D::LinearTable1D(const int numX,
                                             const char *id,
                                             const real8 *x,
                                             const real8 *fx)
   : DataTable1D(numX, id, x, fx)
   , m_x0(new real8[numX-1])
   , m_xd(new real8[numX-1])
   , m_fx0(new real8[numX-1])
   , m_fxd(new real8[numX-1])
{
   int len = m_numX - 1;

   for (int ix1 = 0 ; ix1 < len ; ++ix1) {
      int ix2 = ix1+1 ;

      real8 x1  = GetXAt(ix1) ;
      real8 x2  = GetXAt(ix2) ;
      real8 fx1   = GetYAt(ix1) ;
      real8 fx2   = GetYAt(ix2) ;
      real8 delx =  x2 -  x1 ;
      real8 delf = fx2 - fx1 ;


      if (delx != 0.) {
         m_fxd[ix1] = (fx2-fx1)/delx ;
         m_fx0[ix1] = (fx1*x2 - fx2*x1)/delx ;
      }
      else {
         m_fx0[ix1] = 0. ;
         m_fxd[ix1] = 0. ;
      }

      if (delf != 0.) {
         m_xd[ix1] = (x2-x1)/delf ;
         m_x0[ix1] = (x1*fx2 - x2*fx1)/delf ;
      }
      else {
         m_x0[ix1] = 0. ;
         m_xd[ix1] = 0. ;
      }
   }
}

CHAI_HOST_DEVICE LinearTable1D::~LinearTable1D()
{
   if (m_numX > 1) {
      delete [] m_x0 ;
      delete [] m_xd ;
      delete [] m_fx0 ;
      delete [] m_fxd ;
   }
}

CHAI_HOST_DEVICE real8 LinearTable1D::innerEvaluate(real8 xin) const
{
   int ix1 = SearchX(xin) ;
   return m_fx0[ix1] + m_fxd[ix1] * xin ;
}

CHAI_HOST_DEVICE DerivedTable1D::DerivedTable1D(const char *id)
   : Table1D(id)
   , Table::Data()
   , Table::Derived()
{
#ifdef CHAI_DEVICE_COMPILE
   printf("DerivedTable1D::DerivedTable1D POINTER %p\n", this) ;
#endif
}

CHAI_HOST_DEVICE real8 DerivedTable1D::Evaluate(real8 xin) const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("DerivedTable1D::Evaluate POINTER %p\n", this) ;
#endif
   return innerEvaluate(xin) ;
}

CHAI_HOST_DEVICE Table1D::Table1D(const char* id)
   : SimpleTable(id)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("Table1D::Table1D POINTER %p\n", this) ;
#endif
}

CHAI_HOST_DEVICE DataTable1D::DataTable1D(const int numX,
                                         const char *id,
                                         const real8 *x,
                                         const real8 *fx)
   : Table1D(id),
   Table::Data(),
   m_numX((numX == 1) ? 2 : numX),
   m_x(new real8[m_numX]),
   m_fx(new real8[m_numX])
{
   for (int i = 0 ; i < m_numX ; ++i) {
      m_x[i] = x[i] ;
      m_fx[i] = fx[i] ;
   }
}

CHAI_HOST_DEVICE DataTable1D::~DataTable1D()
{
   if (m_numX > 0) {
      delete [] m_x ;
      delete [] m_fx ;
   }
}

CHAI_HOST_DEVICE real8 DataTable1D::Evaluate(real8 xin) const
{
   return innerEvaluate(xin) ;
}

CHAI_HOST_DEVICE int DataTable1D::SearchX(real8 xin) const
{
   return Search(xin, m_x, m_numX) ;
}

CHAI_HOST_DEVICE real8 DataTable1D::GetXAt(int pos) const {
   return m_x[pos] ;
}

CHAI_HOST_DEVICE real8 DataTable1D::GetYAt(int pos) const {
   return m_fx[pos] ;
}

CHAI_HOST StitchTable1D::StitchTable1D(const char* id,
                                         int nt,
                                         Table1D const** tabs,
                                         const real8* bds)
   : DerivedTable1D(id)
   , m_nTables(nt)
   , m_tables(new const Table1D *[nt])
   , m_bds(new real8[nt-1])
{
   for (int i = 0 ; i < nt ; ++i) {
      m_tables[i] = tabs[i] ;
   }

   for (int i = 0 ; i < nt-1 ; ++i) {
      m_bds[i] = bds[i] ;
   }
}

CHAI_HOST_DEVICE StitchTable1D::StitchTable1D(const char* id,
                                             int nt,
                                             chai::managed_ptr<Table1D const>* tabs,
                                             const real8* bds)
   : DerivedTable1D(id)
   , m_nTables(nt)
   , m_tables(new const Table1D *[nt])
   , m_bds(new real8[nt-1])
{
   for (int i = 0 ; i < nt ; ++i) {
      m_tables[i] = tabs[i].get() ;
#ifdef CHAI_DEVICE_COMPILE
      printf("StitchTable1D::StitchTable1D POINTER %p i %d m_tables[i] %p\n", this, i, m_tables[i]) ;
#endif
   }

   for (int i = 0 ; i < nt-1 ; ++i) {
      m_bds[i] = bds[i] ;
   }
}

CHAI_HOST_DEVICE StitchTable1D::~StitchTable1D()
{
   if (m_nTables > 0) {
      delete [] m_tables ;
   }

   if (m_nTables > 1) {
      delete [] m_bds ;
   }
}

CHAI_HOST_DEVICE real8 StitchTable1D::innerEvaluate(real8 xin) const
{
   real8 xv = xin ;
   int i=0 ;
   while ( (i < m_nTables-1) && (xv >= m_bds[i]) ) {
      i++ ;
   }
#ifdef CHAI_DEVICE_COMPILE
   printf("StitchTable1D::innerEvaluate POINTER %p i %d m_tables[i] %p\n", this, i, m_tables[i]) ;
#endif
   return m_tables[i]->Evaluate(xv) ;
}

CHAI_HOST_DEVICE ComputedTable1D::
ComputedTable1D(const char* id, const Table1D * f, real8 min, real8 max)
   : DerivedTable1D(id), Table::Compute(),
   m_table(f)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("ComputedTable1D::ComputedTable1D POINTER %p m_table %p\n", this, m_table) ;
#endif
}

CHAI_HOST_DEVICE YofXfromRTTable1D::YofXfromRTTable1D(const char* id, Table1D const * f,
                                                     real8 xlow,
                                                     real8 xhigh)
   : ComputedTable1D(id, f, xlow, xhigh)
{
#ifdef CHAI_DEVICE_COMPILE
   printf("YofXfromRTTable1D::YofXfromRTTable1D POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
   real8 r_min = m_table->Evaluate(xlow) ;
}

CHAI_HOST_DEVICE real8 YofXfromRTTable1D::RootFromBaseX(real8 t, const real8 *vals) const
{
#ifdef CHAI_DEVICE_COMPILE
   printf("YofXfromRTTable1D::RootFromBaseX POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
   return m_table->Evaluate(t) ;
}

CHAI_HOST_DEVICE real8 YofXfromRTTable1D::innerEvaluate(real8 xin) const
{
   real8 xv = xin ;
#ifdef CHAI_DEVICE_COMPILE
   printf("YofXfromRTTable1D::innerEvaluate POINTER %p m_table %p <<< CHECK THESE POINTERS\n", this, m_table) ;
#endif
   real8 t = BaseXFromNewX(&xv) ;
   return t ;
}

CHAI_HOST_DEVICE RofTfromXYTable1D::RofTfromXYTable1D(const char* id, Table1D const * f,
                                                     real8 xlow,
                                                     real8 xhigh)
   : ComputedTable1D(id, f, xlow, xhigh)

{
#ifdef CHAI_DEVICE_COMPILE
   printf("RofTfromXY::RofTfromXYTable1D POINTER %p m_table %p\n", this, m_table) ;
#endif

   m_table->Evaluate(xlow) ;
}


CHAI_HOST_DEVICE real8 RofTfromXYTable1D::RootFromBaseX(real8 xnew, const real8 *vals) const
{
   return m_table->Evaluate(xnew) ;
}

CHAI_HOST_DEVICE real8 RofTfromXYTable1D::innerEvaluate(real8 xin) const
{
   real8 tv = xin ;
   real8 x = BaseXFromNewX(&tv) ;
   return x ;
}

int main(int, char**) {
   chai::managed_ptr<Table1D const> tabArray[6] ;
   {
      chai::ManagedArray<real8> x(2) ;
      chai::ManagedArray<real8> fx(2) ;
      real8 *x_host = x.data() ;
      real8 *fx_host = fx.data() ;
      x_host[0] = 0.0 ;
      x_host[1] = 180.0 ;
      fx_host[0] = 4.0 ;
      fx_host[1] = 4.0 ;

      tabArray[0] = chai::make_managed<LinearTable1D>(2, "ul1_rt", chai::unpack(x), chai::unpack(fx));
      x.free() ;
      fx.free() ;
   }
   {
      const char * id = "mesh_ul1_rt_up" ;
      real8 bp[2] = { 90.0, 180.0 } ;

      chai::managed_ptr<const Table1D> tabArray_0 = tabArray[0];

      tabArray[1] = chai::make_managed<YofXfromRTTable1D>(id, chai::unpack(tabArray_0), bp[0], bp[1]) ;
   }
   {
      chai::ManagedArray<real8> x(2) ;
      chai::ManagedArray<real8> fx(2) ;
      real8 *x_host = x.data() ;
      real8 *fx_host = fx.data() ;
      x_host[0] = -20.0 ;
      x_host[1] = 20.0 ;
      fx_host[0] = 4.0 ;
      fx_host[1] = 4.0 ;

      tabArray[2] = chai::make_managed<LinearTable1D>(2, "ul1_yx", chai::unpack(x), chai::unpack(fx));
      x.free() ;
      fx.free() ;
   }
   {
      const char * id = "mesh_ul1_up" ;

      chai::ManagedArray<chai::managed_ptr<const Table1D> > host_device_temp(2) ;
      chai::managed_ptr<const Table1D> *host_temp = host_device_temp.data() ;
      host_temp[0] = tabArray[1] ;
      host_temp[1] = tabArray[2] ;

      chai::ManagedArray<real8> host_device_bp(1) ;
      real8 *host_bp = host_device_bp.data() ;
      host_bp[0] = 2.449293593598294706e-16 ;

      tabArray[3] = chai::make_managed<StitchTable1D>(id, 2, chai::unpack(host_device_temp), chai::unpack(host_device_bp)) ;
      host_device_temp.free() ;
      host_device_bp.free() ;
   }
   {
      chai::ManagedArray<real8> x(2) ;
      chai::ManagedArray<real8> fx(2) ;
      real8 *x_host = x.data() ;
      real8 *fx_host = fx.data() ;
      x_host[0] = 2.5 ;
      x_host[1] = 4.0 ;
      fx_host[0] = 3.17999364001908 ;
      fx_host[1] = 5.0879898240305277 ;

      tabArray[4] = chai::make_managed<LinearTable1D>(2, "mesh_cut_up_right_center_1", chai::unpack(x), chai::unpack(fx));
      x.free() ;
      fx.free() ;
   }
   {
      const char * id = "mesh_ul1_up_right_1" ;

      chai::ManagedArray<chai::managed_ptr<const Table1D> > host_device_temp(2) ;
      chai::managed_ptr<const Table1D> *host_temp = host_device_temp.data() ;
      host_temp[0] = tabArray[3] ;
      host_temp[1] = tabArray[4] ;

      chai::ManagedArray<real8> host_device_bp(1) ;
      real8 *host_bp = host_device_bp.data() ;
      host_bp[0] = 2.5 ;

      tabArray[5] = chai::make_managed<StitchTable1D>(id, 2, chai::unpack(host_device_temp), chai::unpack(host_device_bp)) ;

      host_device_bp.free() ;
      host_device_temp.free() ;
   }
   {
      const char * id = "mesh_cut_up_right_1_rt" ;
      real8 bp[2] = { 0.0, 20.0 } ;
      chai::managed_ptr<const Table1D> tabArray_0 = tabArray[5];

      chai::make_managed<RofTfromXYTable1D>(id, chai::unpack(tabArray_0), bp[0], bp[1]) ;
   }

   printf("SUCCESS\n") ;
   return 0;
}

