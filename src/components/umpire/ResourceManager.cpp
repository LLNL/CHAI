#include "ResourceManager.hpp"

namespace umpire {

ResourceManager* ResourceManager::s_resource_manager_instance = nullptr;

ResourceManager* ResourceManager::getResourceManager() {
  if (!s_resource_manager_instance) {
    s_resource_manager_instance = new ResourceManager();
  }

  return s_resource_manager_instance;
}

ResourceManager::ResourceManager() {
}

size_t ResourceManager::getAvailableMemory() {
  return -1;
}


}
