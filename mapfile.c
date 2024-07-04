#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

float* mmap_file(const char* filename, size_t size) {
    int fd = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("open");
        return NULL;
    }
    if (ftruncate(fd, size) == -1) {
        perror("ftruncate");
        close(fd);
        return NULL;
    }
    float* map = (float*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return NULL;
    }
    close(fd);
    return map;
}

void unmap_file(float* map, size_t size) {
    if (munmap(map, size) == -1) {
        perror("munmap");
    }
}
