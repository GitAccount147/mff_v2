#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
typedef struct words
{
	char* symbols;
	int length;
	struct words* next;
} word;
word* add_word(word* last)
{
	word* new_word = malloc(sizeof(word));
	new_word->next = NULL;
	last->next = new_word;
	return new_word;
}
int read_n(FILE* input_file)
{
	int n = 0;
	char c = fgetc(input_file);
	while (c == ' ')
	{
		c = fgetc(input_file);
	}
	while (c != '\n' && c != ' ')
	{
		n = (n * 10) + c - '0';
		c = fgetc(input_file);
	}
	while (c == ' ')
	{
		c = fgetc(input_file);
	}
	return n;
}
char* get_line(FILE* input_file)
{
	char* line = malloc(103 * sizeof(char));
	char c = fgetc(input_file);
	while (c == ' ')
	{
		c = fgetc(input_file);
	}
	int i = 0;
	while (c != '\n' && c!= EOF)
	{
		while (c != ' ' && c != '\n' && c != EOF)
		{
			line[i] = c;
			i++;
			c = fgetc(input_file);
		}
		line[i] = ' ';
		i++;
		while (c == ' ')
		{
			c = fgetc(input_file);
		}
	}
	line[i] = '\n';
	if (c == EOF)
	{
		line[i + 1] = 0;
	}
	else
	{
		line[i + 1] = 1;
	}
	return line;
}
word* fill_new_line(word* last_word, char* line)
{
	return last_word;
}
int main()
{
	FILE* input_file = fopen("testin.txt", "r");
	FILE* output_file = fopen("testout.txt", "w");
	
	int n = read_n(input_file);

	word* head = malloc(sizeof(word));
	head->next = NULL;
	head->length = 0;
	head->symbols = NULL;
	word* last_word = head;


	char* line;
	int i;
	int end = 0;
	while (!end)
	{
		i = 0;
		line = get_line(input_file);
		last_word = fill_new_line(head, line);
		while (line[i] != '\n')
		{
			printf("(%c)", line[i]);
			i++;
			if (line[i] == '\n' && !line[i + 1])
			{
				end = 1;
				printf("(EOF)");
			}
		}
		printf("\n");
	}

	fclose(input_file);
	fclose(output_file);

	return 0;
}