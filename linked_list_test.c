#include <stdio.h>

#include "linked_list.h"

int main()
{
    linked_list *head = list_create(0);

    for(int i=1; i<20000; i++) append(head, (void *)i);

    append_no_repeat(head, head); //将不会真正向链表中添加节点

    // for(int i=0; i<10; i++) printf("%lu\n", list_index(head, i));

    printf("index:%lu\n", list_in(head, (void *)10));
    printf("list len:%lu\n", (unsigned long)list_len(head));
}