#include <stdio.h>
long rec(int *fail)
{
	long a = getchar();
	long b = getchar();
	if ((a == '+' || a == '-' || a == '*' || a == '/') && b == ' ')
	{
		long left = rec(fail);
		long right = rec(fail);
		if (a == '+')
		{
			return left + right;
		}
		if (a == '-')
		{
			return left - right;
		}
		if (a == '*')
		{
			return left * right;
		}
		if (a == '/')
		{
			if (right == 0)
			{
				*fail = 1;
				return 0;
			}
			else
			{
				return left / right;
			}
		}
	}
	else
	{
		if (a == '-')
		{
			b -= 48;
			int digit_char = getchar();
			while (digit_char >= 48 && digit_char <= 57)
			{
				b = (b * 10) + (digit_char - 48);
				digit_char = getchar();
			}
			return (-1) * b;
		}
		else
		{
			a -= 48;
			if (b != ' ' && b != 10)
			{
				b -= 48;
				a = (a * 10) + b;
				int digit_char = getchar();
				while (digit_char >= 48 && digit_char <= 57)
				{
					a = (a * 10) + (digit_char - 48);
					digit_char = getchar();
				}
			}
			return a;
		}
	}
}
int main()
{
	int fail = 0;
	long ret = rec(&fail);
	if (fail == 0)
	{
		printf("%ld", ret);
	}
	else
	{
		printf("CHYBA");
	}
	return 0;
}