#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
typedef struct morse_cells
{
	morse_cells* dot;
	morse_cells* dash;
	char* value;
	int value_len;
}morse_cell;
morse_cell* add_cell()
{
	morse_cell* new_cell = new morse_cell;
	new_cell->dot = NULL;
	new_cell->dash = NULL;
	new_cell->value = NULL;
	return new_cell;
}
morse_cell* create_tree()
{
	morse_cell* head = new morse_cell;
	head->dot = NULL;
	head->dash = NULL;
	head->value = NULL;

	FILE* dictionary = fopen("morse.txt", "r");
	char c = fgetc(dictionary);
	while (c != EOF)
	{
		while (c == ' ')
		{
			c = fgetc(dictionary);
		}
		if (c == EOF) // asi by nemelo nastat (~ na konci souboru neni '\n' ale rovnou EOF)
		{
			fclose(dictionary);
			return head;
		}
		morse_cell* current_cell = head;
		while (c == '.' || c == '-')
		{
			if (c == '.')
			{
				if (current_cell->dot == NULL)
				{
					current_cell->dot = add_cell();
				}
				current_cell = current_cell->dot;
			}
			else
			{
				if (current_cell->dash == NULL)
				{
					current_cell->dash = add_cell();
				}
				current_cell = current_cell->dash;
			}
			c = fgetc(dictionary);
		}

		while (c == ' ')
		{
			c = fgetc(dictionary);
		}

		char* value = new char[3];
		int value_len = 0;
		while (c != ' ' && c != '\n' && c != EOF)
		{
			if (c >= 'A' && c <= 'Z')
			{
				c = c + 'a' - 'A';
			}
			value[value_len] = c;
			value_len++;
			c = fgetc(dictionary);
		}
		current_cell->value = value;
		current_cell->value_len = value_len;

		while (c == ' ' || c == '\n')
		{
			c = fgetc(dictionary);
		}
		if (c == EOF)
		{
			fclose(dictionary);
			return head;
		}
	}
	fclose(dictionary);
	return head;
}
void decode(morse_cell* head)
{
	FILE* input_file = fopen("vstup.txt", "r");
	FILE* output_file = fopen("vystup.txt", "w");

	char c = fgetc(input_file);
	//printf("(1)");
	morse_cell* current_cell = head;
	while (c != EOF)
	{
		while (c == '.' || c == '-')
		{
			if (c == '.')
			{
				if (current_cell->dot != NULL)
				{
					current_cell = current_cell->dot;
					c = fgetc(input_file);
					//printf("(2)");
				}
				else
				{
					while (c == '.' || c == '-')
					{
						c = fgetc(input_file);
						//printf("(3)");
					}
					current_cell = head;
				}
			}
			else if (c == '-')
			{
				if (current_cell->dash != NULL)
				{
					current_cell = current_cell->dash;
					c = fgetc(input_file);
					//printf("(4)");
				}
				else
				{
					while (c == '.' || c == '-')
					{
						c = fgetc(input_file);
						//printf("(5)");
					}
					current_cell = head;
				}
			}
			if (c == '/')
			{
				if (current_cell != head && current_cell->value != NULL)
				{
					for (int i = 0; i < current_cell->value_len; i++)
					{
						fputc(current_cell->value[i], output_file);
						//printf("(write:%c)", current_cell->value[i]);
					}
				}
				c = fgetc(input_file);
				//printf("(6)");
				if (c == '/')
				{
					fputc(' ', output_file);
					//printf("(write: )");
					c = fgetc(input_file);
					//printf("(7)");
				}
				while (c == '\n')
				{
					fputc('\n', output_file);
					//printf("(write:EL)");
					c = fgetc(input_file);
					//printf("(8)");
				}
				current_cell = head;
			}
		}
	}
	fclose(input_file);
	fclose(output_file);
}
void test(morse_cell* cur)
{
	if (cur->value != NULL)
	{
		for (int i = 0; i < cur->value_len; i++)
		{
			printf("%c", cur->value[i]);
		}
	}
	if (cur->dot != NULL)
	{
		printf("(.)");
		test(cur->dot);
	}
	if (cur->dash != NULL)
	{
		printf("(-)");
		test(cur->dash);
	}
	printf("(_)");
}
int main()
{
	morse_cell* head = create_tree();
	//test(head);
	decode(head);
	return 0;
}
