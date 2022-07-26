#ifndef __LINKED_LIST__
#define __LINKED_LIST__

#include <stdint.h>
#include <stdbool.h>

#define NOT_IN_LIST UINT64_MAX

struct linked_list
{
    void *p;
    struct linked_list *next;
};

typedef struct linked_list linked_list;

linked_list *list_create(void *p);
void list_free(linked_list *head_node);
linked_list *list_tail(linked_list *head);
void append(linked_list *list, void *p);
bool append_no_repeat(linked_list *head, void *p);
linked_list *list_index(linked_list *list, size_t i);
uint64_t list_in(linked_list *head, void *p);
size_t list_len(linked_list *head);

#endif