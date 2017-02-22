#ifndef CHAI_ResourceManager_HPP
#define CHAI_ResourceManager_HPP

namespace chai {

struct PointerRecord 
{
  void * m_host_ptr;
  void * m_device_ptr;

  size_t m_size;

  bool m_host_touched;
  bool m_device_touched;
};


class ResourceManager
{
  static ResourceManager* getResourceManager();

  

  protected:

  ResourceManager();

  private:

  static ResourceManager* s_resource_manager_instance;

  std::map<void *, PointerRecord> m_pointer_map;
  std::set<void *> m_accessed_variables;
};

#endif
