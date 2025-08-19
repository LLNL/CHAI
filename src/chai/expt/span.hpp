#pragma once

#include <cstddef>
#include <type_traits>

namespace custom {

template <typename T>
class span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    // Constructors
    __host__ __device__ constexpr span() noexcept : data_(nullptr), size_(0) {}
    
    __host__ __device__ constexpr span(pointer ptr, size_type count) noexcept
        : data_(ptr), size_(count) {}
    
    template <size_t N>
    __host__ __device__ constexpr span(element_type (&arr)[N]) noexcept
        : data_(arr), size_(N) {}
    
    // Element access
    __host__ __device__ constexpr reference operator[](size_type idx) const noexcept {
        return data_[idx];
    }
    
    __host__ __device__ constexpr reference front() const noexcept {
        return data_[0];
    }
    
    __host__ __device__ constexpr reference back() const noexcept {
        return data_[size_ - 1];
    }
    
    __host__ __device__ constexpr pointer data() const noexcept {
        return data_;
    }
    
    // Iterators
    __host__ __device__ constexpr iterator begin() const noexcept {
        return data_;
    }
    
    __host__ __device__ constexpr iterator end() const noexcept {
        return data_ + size_;
    }
    
    // Capacity
    __host__ __device__ constexpr bool empty() const noexcept {
        return size_ == 0;
    }
    
    __host__ __device__ constexpr size_type size() const noexcept {
        return size_;
    }
    
    __host__ __device__ constexpr size_type size_bytes() const noexcept {
        return size_ * sizeof(element_type);
    }
    
    // Subviews
    __host__ __device__ constexpr span<element_type> first(size_type count) const {
        return {data_, count};
    }
    
    __host__ __device__ constexpr span<element_type> last(size_type count) const {
        return {data_ + (size_ - count), count};
    }
    
    __host__ __device__ constexpr span<element_type> subspan(size_type offset, size_type count) const {
        return {data_ + offset, count};
    }

private:
    pointer data_;
    size_type size_;
};

// Deduction guides
template <typename T, std::size_t N>
span(T (&)[N]) -> span<T>;

template <typename T>
span(T*, std::size_t) -> span<T>;

} // namespace custom