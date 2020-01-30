
#ifndef CHAI_ArrayConfig_HPP
#define CHAI_ArrayConfig_HPP

namespace chai {

namespace config {

  enum class Codec {
    none,
    zfp
  };

  namespace storage {
#if CHAI_HAS_ZFP
    template <typename T, size_t = T::codec_dims>
    struct zfp;

    template <typename T>
    struct zfp<T,1> {
      using array     = ::zfp::array1<T::value_type>;
      using pointer   = array::pointer;
      using reference = array::reference;
    }

    template <typename T>
    struct zfp<T,2> {
      using array     = ::zfp::array2<T::value_type>;
      using pointer   = array::pointer;
      using reference = array::reference;
    }

    template <typename T>
    struct zfp<T,3> {
      using array     = ::zfp::array3<T::value_type>;
      using pointer   = array::pointer;
      using reference = array::reference;
    }
#endif
  }

  struct ArrayDesc {
    static constexpr chai::config::Codec codec = chai::config::Codec::none;
  };

  template < typename ArrayT, bool = std::is_base_of<ArrayDesc, ArrayT>::value >
  struct ConstArrayDesc;

  template < typename ArrayT>
  struct ConstArrayDesc<ArrayT, true> : ArrayT {
    using value_type = const typename ArrayT::value_type;
  };

  template < typename T >
  struct ConstArrayDesc<T, false> : ArrayDesc {
    using value_type = const T;
  };

  // Value Type

  template <
    typename T,
    bool uses_config_type = std::is_base_of<ArrayDesc, T>::value,
    Codec = std::conditional<uses_config_type, T, ArrayDesc>::type::codec
  >
  struct Types;

  template <typename ValueT>
  struct Types<ValueT, false, Codec::none> {
    using value_type               = ValueT;
    using value_type_const         = ValueT const;
    using value_type_non_const     = typename std::remove_const<ValueT>::type;

    using pointer_type             = ValueT *;
    using pointer_type_const       = ValueT const *;
    using pointer_type_non_const   = typename std::remove_const<ValueT>::type *;

    using reference_type           = ValueT &;
    using reference_type_const     = ValueT const &;
    using reference_type_non_const = typename std::remove_const<ValueT>::type &;
  };

  template <typename ArrCfgT>
  struct Types<ArrCfgT, true, Codec::none> {
    using value_type               = typename ArrCfgT::value_type;
    using value_type_const         = typename ArrCfgT::value_type const;
    using value_type_non_const     = typename std::remove_const<typename ArrCfgT::value_type>::type;

    using pointer_type             = typename ArrCfgT::value_type *;
    using pointer_type_const       = typename ArrCfgT::value_type const *;
    using pointer_type_non_const   = typename std::remove_const<typename ArrCfgT::value_type>::type *;

    using reference_type           = typename ArrCfgT::value_type &;
    using reference_type_const     = typename ArrCfgT::value_type const &;
    using reference_type_non_const = typename std::remove_const<typename ArrCfgT::value_type>::type &;
  };

#if CHAI_HAS_ZFP
  template <typename ArrCfgT>
  struct Types<ArrCfgT, true, Codec::zfp> {
    using value_type               = typename ArrCfgT::value_type;
    using value_type_const         = typename ArrCfgT::value_type const;
    using value_type_non_const     = typename std::remove_const<typename ArrCfgT::value_type>::type;

    using pointer_type             = typename storage::zfp<ArrCfgT>::pointer;
    using pointer_type_const       = void; // TODO const pointer for ZFP?
    using pointer_type_non_const   = void; // TODO non const pointer for ZFP?

    using reference_type           = typename storage::zfp<ArrCfgT>::reference;
    using reference_type_const     = void; // TODO const reference for ZFP?
    using reference_type_non_const = void; // TODO non const reference for ZFP?
  };
#endif
}

}


#endif  // CHAI_ArrayConfig_HPP

