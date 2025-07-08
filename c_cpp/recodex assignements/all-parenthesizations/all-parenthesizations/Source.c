#include <stdio.h>
int read_positive_int()
{
	int number = 0;
	int digit_char = getchar();
	while (digit_char >= '0' && digit_char <= '9')
	{
		number = (number * 10) + (digit_char - '0');
		digit_char = getchar();
	}
	return number;
}
void rec(int closed, int opened, int unused, int depth, int n, int *log)
{
	if (unused != 0)
	{
		log[depth] = 1;
		rec(closed, opened + 1, unused - 1, depth + 1, n, log);
	}
	if (opened != 0)
	{
		log[depth] = 0;
		rec(closed + 1, opened - 1, unused, depth + 1, n, log);
	}
	if (closed == n)
	{
		for (int i = 0; i < (2 * n); i++)
		{
			if (log[i])
			{
				printf("(");
			}
			else
			{
				printf(")");
			}
		}
		printf("\n");
	}
}
int main()
{
	int n = read_positive_int();
	int log[50];
	rec(0, 0, n, 0, n, log);
	return 0;
}