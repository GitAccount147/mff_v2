#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
typedef struct tiles
{
	int row;
	int column;
	int depth;
	struct tiles* next;
} tile;
tile* add(tile* last_tile, int row, int column, int depth)
{
	tile* new_tile = malloc(sizeof(tile));
	new_tile->row = row;
	new_tile->column = column;
	new_tile->depth = depth;
	new_tile->next = NULL;
	last_tile->next = new_tile;
	return new_tile;
}
void pop(tile* head, int* row, int* column, int* depth)
{
	tile* temp = head->next;
	*row = temp->row;
	*column = temp->column;
	*depth = temp->depth;
	head->next = temp->next;
	free(temp);
}
int read_number(FILE* input_file)
{
	char c = fgetc(input_file);
	int number = 0;
	while (c != '\n')
	{
		number = (10 * number) + (c - '0');
		c = fgetc(input_file);
	}
	return number;
}
int vectors[8][2] = { {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1} };
int* create_board(FILE* input_file, int rows, int columns, int* start_r, int* start_c, int* finish_r, int* finish_c)
{
	int* board = malloc(rows * columns * sizeof(int));
	char c;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			c = fgetc(input_file);
			if (c == '.')
			{
				board[i * columns + j] = -1;
			}
			else if (c == 'X')
			{
				board[i * columns + j] = -2;
			}
			else if (c == 'S')
			{
				board[i * columns + j] = -1;
				*start_r = i;
				*start_c = j;
			}
			else
			{
				board[i * columns + j] = -1;
				*finish_r = i;
				*finish_c = j;
			}
		}
		fgetc(input_file);
	}
	return board;
}
void color_board(int* board, tile* head, int rows, int columns)
{
	int row, column, depth;
	tile* last_tile = head->next;
	tile* temp;
	while (head->next != NULL)
	{
		if (last_tile == head->next)
		{
			last_tile = head;
		}
		pop(head, &row, &column, &depth);
		//printf("r c d: %d %d %d\n", row, column, depth);
		for (int i = 0; i < 8; i++)
		{
			if (row + vectors[i][0] > -1 && row + vectors[i][0] < rows && column + vectors[i][1] > -1 && column + vectors[i][1] < columns)
			{
				if (board[(row + vectors[i][0]) * columns + column + vectors[i][1]] == -1)
				{
					//printf("check: %d %d\n", row + vectors[i][0], column + vectors[i][1]);
					board[(row + vectors[i][0]) * columns + column + vectors[i][1]] = depth + 1;
					temp = add(last_tile, row + vectors[i][0], column + vectors[i][1], depth + 1);
					last_tile = last_tile->next;
				}
			}
		}
	}
}
int main()
{
	FILE* input_file = fopen("sachovnice.txt", "r");

	int rows = read_number(input_file);
	int columns = read_number(input_file);

	int start_r, start_c, finish_r, finish_c;
	int* board = create_board(input_file, rows, columns, &start_r, &start_c, &finish_r, &finish_c);

	tile* head = malloc(sizeof(tile));
	tile* first = add(head, start_r, start_c, 0);
	board[start_r * rows + start_c] = 0;

	color_board(board, head, rows, columns);

	printf("%d", board[finish_r * columns + finish_c]);

	fclose(input_file);

	return 0;
}