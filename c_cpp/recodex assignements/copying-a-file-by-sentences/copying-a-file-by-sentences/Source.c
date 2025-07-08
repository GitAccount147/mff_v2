#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main()
{
	FILE* old_file = fopen("a.txt", "r");
	FILE* new_file = fopen("b.txt", "w");
	char c = getc(old_file);
	while (c != EOF)
	{
		if (c == '.')
		{
			fputc('.', new_file);
			c = getc(old_file);
			while (c != EOF && (c == '\n' || c == ' '))
			{
				c = getc(old_file);
			}
			if (c != EOF)
			{
				fputc('\n', new_file);
			}
		}
		else if (c == '\n')
		{
			fputc(' ', new_file);
			c = getc(old_file);
		}
		else
		{
			fputc(c, new_file);
			c = getc(old_file);
		}
	}
	fclose(old_file);
	fclose(new_file);
	return 0;
}