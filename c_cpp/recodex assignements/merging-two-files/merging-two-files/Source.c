#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int file_read_int(FILE* input, int* process_status)
{
	char c = fgetc(input);
	while (!(c >= '0' && c<='9') && c!='-' && c!=EOF)
	{
		c = fgetc(input);
	}
	int number = 0;
	int minus = 0;

	if (c != EOF)
	{
		if (c == '-')
		{
			minus = 1;
			c = fgetc(input);
		}
		while (c >= '0' && c <= '9')
		{
			number = (10 * number) + (c - '0');
			c = fgetc(input);
			if (c == EOF)
			{
				*process_status = 1;
			}
		}
		if (minus)
		{
			number = (-1) * number;
		}
		*process_status = 1;
	}
	else
	{
		*process_status = 0;
	}
	return number;
}
void file_write_int(FILE* output, int number, int first)
{
	int digit;
	if (!first)
	{
		fputc(' ', output);
	}
	if (number < 0)
	{
		fputc('-', output);
		number = (-1) * number;
	}
	int highest = 10;
	while (number / highest != 0)
	{
		highest = 10 * highest;
	}
	highest = highest / 10;
	while (highest != 1)
	{
		digit = number / highest;
		fputc('0' + digit, output);
		number = number - (highest * digit);
		highest = highest / 10;
	}
	fputc('0' + number, output);
}
int main()
{
	FILE* input_file_1 = fopen("A1.TXT", "r");
	FILE* input_file_2 = fopen("A2.TXT", "r");
	FILE* output_file = fopen("B.TXT", "w");

	int procces_status = 0;
	
	int empty_file_1 = 1;
	int empty_file_2 = 1;

	int number_1 = file_read_int(input_file_1, &procces_status);
	int file_1_status = procces_status;

	empty_file_1 = procces_status;

	int number_2 = file_read_int(input_file_2, &procces_status);
	int file_2_status = procces_status;

	empty_file_2 = procces_status;

	int first = 1;

	/*
	while (file_1_status || file_2_status)
	{
		if (file_1_status && file_2_status)
		{
			if (number_1 < number_2)
			{
				//printf("{%d}\n", number_1);
				file_write_int(output_file, number_1, first);
				number_1 = file_read_int(input_file_1, &procces_status);
				file_1_status = procces_status;
			}
			else
			{
				//printf("{%d}\n", number_2);
				file_write_int(output_file, number_2, first);
				number_2 = file_read_int(input_file_2, &procces_status);
				file_2_status = procces_status;
			}
		}
		else if (file_1_status)
		{
			//printf("{%d}\n", number_1);
			file_write_int(output_file, number_1, first);
			number_1 = file_read_int(input_file_1, &procces_status);
			file_1_status = procces_status;
		}
		else
		{
			//printf("{%d}\n", number_2);
			file_write_int(output_file, number_2, first);
			number_2 = file_read_int(input_file_2, &procces_status);
			file_2_status = procces_status;
		}
		first = 0;
	}
	*/

	while (empty_file_1 || empty_file_2)
	{
		//printf("{%d %d}", empty_file_1, empty_file_2);
		if (empty_file_1 && empty_file_2)
		{
			if (number_1 < number_2)
			{
				file_write_int(output_file, number_1, first);
				number_1 = file_read_int(input_file_1, &procces_status);
				if (!procces_status)
				{
					empty_file_1 = 0;
				}
			}
			else
			{
				file_write_int(output_file, number_2, first);
				number_2 = file_read_int(input_file_2, &procces_status);
				if (!procces_status)
				{
					empty_file_2 = 0;
				}
			}
		}
		else if (empty_file_1)
		{
			file_write_int(output_file, number_1, first);
			number_1 = file_read_int(input_file_1, &procces_status);
			if (!procces_status)
			{
				empty_file_1 = 0;
			}
		}
		else
		{
			file_write_int(output_file, number_2, first);
			number_2 = file_read_int(input_file_2, &procces_status);
			if (!procces_status)
			{
				empty_file_2 = 0;
			}
		}
		first = 0;
	}

	fclose(input_file_1);
	fclose(input_file_2);
	fclose(output_file);
	
	return 0;
}