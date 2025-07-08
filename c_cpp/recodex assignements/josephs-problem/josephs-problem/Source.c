#include <stdio.h>
int read_and_check_int()
{
	// unsigned nonzero longint (1...2147483647)
	int max[] = {2, 1, 4, 7, 4, 8, 3, 6, 4, 7};
	int digits[10];
	int length = 0;
	int digit_char = getchar();
	while (digit_char > 31)
	{
		if (length == 10)
		{
			//printf("(len>10)");
			return 0;
		}
		if (digit_char >= '0' && digit_char <= '9')
		{
			digits[length] = digit_char - '0';
			digit_char = getchar();
			length++;
		}
		else
		{
			//printf("(non deci char)");
			return 0;
		}
	}
	if (length == 10)
	{
		for (int i = 0; i < 10; i++)
		{
			if (digits[i] > max[i])
			{
				//printf("(>longint)");
				return 0;
			}
			else if(digits[i] < max[i])
			{
				i = 10;
			}
		}
	}
	int first_nonzero_digit = 0;
	for (int i = 0; i < length; i++)
	{
		if (digits[i] != 0)
		{
			first_nonzero_digit = 1;
		}
		if (digits[i] == 0 && first_nonzero_digit == 0)
		{
			//printf("(zerostart)");
			return 0;
		}
	}
	if (length > 0)
	{
		int number = 0;
		for (int i = 0; i < length; i++)
		{
			number = (number * 10) + digits[i];
		}
		return number;
	}
	else
	{
		//printf("(len<1)");
		return 0;
	}
}
int wheres_joseph(int number)
{
	int expo = 1;
	for (int i = 0; i < 31; i++)
	{
		if (number / expo == 0)
		{
			i = 31;
		}
		else if (expo < 1000000000)
		{
			expo = expo * 2;
		}
	}
	if (number > 1073741824)
	{
		return (((number - expo) * 2) + 1);
	}
	else
	{
		return (((number - expo / 2) * 2) + 1);
	}
}
int main()
{
	int input = read_and_check_int();
	if (input)
	{
		printf("%d", wheres_joseph(input));
	}
	else
	{
		printf("ERROR");
	}
	return 0;
}