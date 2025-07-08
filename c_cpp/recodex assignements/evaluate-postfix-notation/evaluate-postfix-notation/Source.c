#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
typedef struct blocks
{
	int value;
	struct blocks* next;
} block;
void add_number(block* head, int value)
{
	block* new_block = malloc(sizeof(block));
	block* temp = head->next;
	head->next = new_block;
	new_block->next = temp;
	new_block->value = value;
}
int apply_operator(block* head, int operator)
{
	if (head->next != NULL && head->next->next != NULL)
	{
		int first_number = head->next->value;
		int second_number = head->next->next->value;
		int number;
		switch (operator)
		{
		case -1:
			number = second_number + first_number;
			break;
		case -2:
			number = second_number - first_number;
			break;
		case -3:
			number = second_number * first_number;
			break;
		case -4:
			number = second_number / first_number;
			break;
		}
		block* temp = head->next->next->next;
		free(head->next->next);
		head->next->next = temp;
		head->next->value = number;
		//printf("{%d}", head->next->value);
		return 1;
	}
	return 0;
}
int get_next(int * end)
{
	int number = 0;
	char c = getchar();
	if (c == EOF)
	{
		*end = 1;
		return -5;
	}
	while (!((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '*' || c == '/'))
	{
		c = getchar();
		if (c == EOF)
		{
			*end = 1;
			return -5;
		}
	}
	if (!(c >= '0' && c <= '9'))
	{
		switch (c)
		{
		case '+':
			//printf("+");
			return -1;
		case '-':
			//printf("-");
			return -2;
		case '*':
			//printf("*");
			return -3;
		case '/':
			//printf("/");
			return -4;
		}
	}
	while (c >= '0' && c <= '9')
	{
		number = 10 * number + c - '0';
		c = getchar();
		//printf("(%d)", number);
		if (c == EOF)
		{
			*end = 1;
			return number;
		}
	}
	return number;
}
int main()
{
	block* head = malloc(sizeof(block));
	head->next = NULL;
	head->value = -5; //not necessary

	int end = 0;
	int current, correct;
	while (!end)
	{
		current = get_next(&end);
		if (current >= 0)
		{
			add_number(head, current);
		}
		else if (current != -5)
		{
			if (!apply_operator(head, current))
			{
				printf("Chyba!");
				return 0;
			}
		}
		/*
		block* iterator = head->next;
		while (iterator != NULL)
		{
			printf("[%d]", iterator->value);
			iterator = iterator->next;
		}
		printf("\n");
		*/
	}
	if (head->next->next == NULL)
	{
		printf("%d", head->next->value);
	}
	else
	{
		printf("Chyba!");
	}
	return 0;
}