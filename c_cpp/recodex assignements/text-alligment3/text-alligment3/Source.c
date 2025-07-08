#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
typedef struct words
{
	char* symbols;
	int length;
	struct words* next;
} word;
void add_word(word* last)
{
	word* new_word = malloc(sizeof(word));
	new_word->next = NULL;
	last->next = new_word;
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
void attach_word(word* last_word, char* word, int length)
{
	char* only_chars = malloc(length * sizeof(char));
	for (int i = 0; i < length; i++)
	{
		only_chars[i] = word[i];
	}
	add_word(last_word);
	last_word->next->length = length;
	last_word->next->symbols = only_chars;
}
int read_word(FILE* input_file, word* last_word, int* reached_end)
{
	char c = fgetc(input_file);
	while (c == ' ' || c == '\n')
	{
		c = fgetc(input_file);
	}
	if (c == EOF)
	{
		*reached_end = 1;
		return 0;
	}
	char word[100];
	int length = 0;
	while (c != ' ' && c != '\n')
	{
		word[length] = c;
		c = fgetc(input_file);
		length++;
		if (c == EOF)
		{
			*reached_end = 1;
			attach_word(last_word, &word, length);
			return length;
		}
	}
	attach_word(last_word, &word, length);
	return length;
}
word* write_line(FILE* output_file, word* last_word, int reached_end, int word_count, int n, int char_count, int current_word_length)
{
	int words_placed = 0;
	if (word_count != 1 && !reached_end)
	{
		int spaces_per_place = (n - char_count - word_count + 1) / (word_count - 1);
		int extra_spaces = n - ((word_count - 1) * (spaces_per_place + 1)) - char_count;
		//printf("(%d %d %d %d %d)", word_count, n, char_count, spaces_per_place, extra_spaces);
		while (last_word->next != NULL)
		{
			for (int i = 0; i < last_word->length; i++)
			{
				fputc(last_word->symbols[i], output_file);
				//printf("%c", last_word->symbols[i]);
			}
			free(last_word->symbols);
			if (words_placed != word_count - 1)
			{
				for (int i = 0; i < spaces_per_place + 1 + (words_placed < extra_spaces); i++)
				{
					fputc(' ', output_file);
					//printf("@");
				}
			}
			words_placed++;
			word* temp = last_word;
			last_word = last_word->next;
			free(temp);
		}
		fputc('\n', output_file);
		//printf("#\n");
	}
	else if(reached_end)
	{
		while (last_word != NULL)
		{
			if (words_placed)
			{
				fputc(' ', output_file);
				//printf("@");
			}
			for (int i = 0; i < last_word->length; i++)
			{
				fputc(last_word->symbols[i], output_file);
				//printf("%c", last_word->symbols[i]);
			}
			free(last_word->symbols);
			words_placed++;
			word* temp = last_word;
			last_word = last_word->next;
			free(temp);
		}
	}
	else
	{
		while (last_word->next != NULL)
		{
			if (words_placed)
			{
				fputc(' ', output_file);
				//printf("@");
			}
			for (int i = 0; i < last_word->length; i++)
			{
				fputc(last_word->symbols[i], output_file);
				//printf("%c", last_word->symbols[i]);
			}
			free(last_word->symbols);
			words_placed++;
			word* temp = last_word;
			last_word = last_word->next;
			free(temp);
		}
		fputc('\n', output_file);
		//printf("#\n");
	}
	return last_word;
}
int main()
{
	FILE* input_file = fopen("odst.in", "r");
	FILE* output_file = fopen("odst.out", "w");

	int n = read_n(input_file);

	word* head = malloc(sizeof(word));
	head->next = NULL;
	head->length = 0;
	head->symbols = NULL;
	word* last_word = head;

	int reached_end = 0;
	int word_count = 0;
	int char_count = 0;
	int current_word_length = 0;

	while (!reached_end)
	{
		while (char_count + word_count - 1 <= n && !reached_end)
		{
			current_word_length = read_word(input_file, last_word, &reached_end);
			if (current_word_length != 0)
			{
				last_word = last_word->next;
				char_count = char_count + current_word_length;
				word_count++;
			}
		}
		head->next = write_line(output_file, head->next, reached_end, word_count - 1, n, char_count - current_word_length, current_word_length);
		word_count = 1;
		char_count = current_word_length;
	}

	fclose(input_file);
	fclose(output_file);

	return 0;
}