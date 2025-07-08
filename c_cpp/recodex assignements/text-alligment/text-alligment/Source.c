#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
typedef struct symbols
{
	char value;
	struct symbols* next;
} symbol;
typedef struct words
{
	struct symbol* first_char;
	struct words* next;
} word;
word* add_word(word* last)
{
	word* new_word = malloc(sizeof(word));
	//printf("&%p&\n", new_word);
	new_word->next = NULL;
	last->next = new_word;
	//printf("&%p&\n", last->next);
	return new_word;
}
symbol* add_symbol(symbol* last, char value)
{
	symbol* new_symbol = malloc(sizeof(symbol));
	new_symbol->next = NULL;
	new_symbol->value = value;
	last->next = new_symbol;
	printf("[%c]", value);
	return new_symbol;
}
int read_word(FILE* input_file, word* last_word, int* reached_end)
{
	int word_len = 0;
	char c = fgetc(input_file);
	if (c == EOF)
	{
		reached_end = 1;
		return word_len;
	}
	while (c == ' ' || c == '\n')
	{
		c = fgetc(input_file);
		if (c == EOF)
		{
			return -1;
		}
	}
	printf("{predchudce %p}\n", last_word);
	word* temp = add_word(last_word);
	temp = last_word;
	printf("{napojeni: %p}\n", temp->next);
	symbol* last_symbol = last_word->next->first_char;
	while (c != ' ' && c != '\n')
	{
		word_len++;
		last_symbol = add_symbol(last_symbol, c);
		printf("insert:%c, \n", last_symbol->value);
		printf("(%c)", c);
		last_symbol = last_symbol->next;
		c = fgetc(input_file);
		if (c == EOF)
		{
			reached_end = 1;
			return word_len;
		}
	}
	symbol* test = last_word->next->first_char;
	while (test != NULL)
	{
		printf("val:%d\n", test->value);
		test = test->next;
	}
	return word_len;
}
word* write_word(FILE* output_file, word* last_word)
{
	symbol* last_symbol = last_word->first_char;
	symbol* temp;
	word* new_word = last_word->next;
	free(last_word);
	while (last_symbol != NULL)
	{
		printf("(%c)", last_symbol->value);
		fputc(last_symbol->value, output_file);
		temp = last_symbol;
		last_symbol = last_symbol->next;
		free(temp);
	}
	return new_word;
}
void write_line(FILE* output_file, word* last_word, int last_line, int word_count, int n, int char_len)
{
	int extra_spaces = n - char_len - word_count + 1;
	int spaces_per_place = extra_spaces / (word_count - 1);
	int additional_spaces = extra_spaces - ((word_count - 1) * spaces_per_place);
	if (!last_line)
	{
		for (int i = 0; i < word_count; i++)
		{
			last_word = write_word(output_file, last_word);
			if (i != word_count - 1)
			{
				if (i < additional_spaces)
				{
					for (int j = 0; j < spaces_per_place + 1; j++)
					{
						fputc(' ', output_file);
					}
				}
				else
				{
					for (int j = 0; j < spaces_per_place; j++)
					{
						fputc(' ', output_file);
					}
				}
			}
			fputc('\n', output_file);
		}
	}
	else
	{
		for (int i = 0; i < word_count; i++)
		{
			last_word = write_word(output_file, last_word);
			if (i != word_count - 1)
			{
				fputc(' ', output_file);
			}
		}
	}
}
int main()
{
	int n = 25;

	word* head = malloc(sizeof(word));
	head->next = NULL;
	head->first_char = NULL;

	word* last_word = head;
	printf("head:%p\n", head);
	
	///*
	FILE* input_file = fopen("testin.txt", "r");
	FILE* output_file = fopen("testout.txt", "w");

	int word_count = 0;
	int char_len = 0;
	int reached_end = 0;
	int current_word_len;

	while (!reached_end)
	{
		while (char_len + word_count - 1 <= n && !reached_end)
		{
			current_word_len = read_word(input_file, last_word, &reached_end);
			printf("!%p, next:%p!\n", last_word, last_word->next);
			last_word = last_word->next;
			if (current_word_len != 0)
			{
				char_len = char_len + current_word_len;
				word_count++;
			}
			printf("\n ******* \n");
		}
		printf("\n ========= \n");
		if (reached_end)
		{
			write_line(output_file, head->next, 1, word_count, n, char_len);
		}
		else
		{
			write_line(output_file, head->next, 0, word_count, n, char_len);
		}
		word_count = 0;
		char_len = 0;
	}
	fclose(input_file);
	fclose(output_file);
	//*/
	return 0;
}