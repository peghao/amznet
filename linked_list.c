/**
 * @brief 实现了单向链表及其基本操作
 * 
 * @note 注意：链表长度大于20k时，对链表的操作会变得十分缓慢，使用时应注意这一点（使用单向有环的链表，会改善一部分操作(如append())的性能，但性能改进以后再说吧）
 * 
 * @date 2022-07-22
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "linked_list.h"

/**
 * @brief 创建一个链表节点
 * 
 * @param init_data 
 * @return linked_list* 
 */
linked_list *list_create(void *p)
{
    linked_list *head_node = (linked_list *)malloc(sizeof(linked_list));
    head_node->p = p;
    head_node->next = NULL;
    return head_node;
}

void list_free(linked_list *head_node)
{
    linked_list *a = head_node;
    linked_list *b = head_node->next;
    while (b != NULL)
    {
        a->next = NULL;
        free(a);
        a = b; b = b->next;
    }
    free(a);
}

linked_list *list_tail(linked_list *head)
{
    if(head == NULL) return NULL;
    linked_list *tail_node = head;
    for(; tail_node->next != NULL; tail_node = tail_node->next);
    return tail_node;
}

void append(linked_list *list, void *p)
{
    linked_list *node = (linked_list *)malloc(sizeof(linked_list));
    node->p = p;
    node->next = NULL;
    list_tail(list)->next = node;
}

/**
 * @brief 向链表中追加元素，但如果链表中已经存在该元素则不再添加
 * 
 * @param head 
 * @param p 要添加的元素
 * @return true 
 * @return false 链表中已存在，实际上没有添加
 */
bool append_no_repeat(linked_list *head, void *p)
{
    for(;;)
    {
        if(head->p == p) return false;
        if(head->next == NULL)
        {
            linked_list *tail_node = list_create(p);
            head->next = tail_node;
            return true;
        }
        head = head->next;
    }
}

linked_list *list_index(linked_list *list, size_t i)
{
    for(size_t j=0; j<i; j++)
    {
        list = list->next;
        if(list == NULL) printf("index error: index out of bound!\n"), exit(-1);
    }
    return list;
}

uint64_t list_in(linked_list *head, void *p)
{
    for(size_t i=0; ;i++)
    {
        if(head->p == p) return i;
        if(head->next == NULL) return NOT_IN_LIST;
        head = head->next;
    }
}

size_t list_len(linked_list *head)
{
    for(size_t i=0; ;i++)
    {
        if(head->next == NULL) return i+1;
        head = head->next;
    }
}
