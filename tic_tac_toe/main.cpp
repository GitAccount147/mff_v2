/*
 5-in-a-row (v2) Josef Sykora
 viz.:
 "Pišqorky 2 (5/4)
 Chytřejší implementace hry piškvorky – prohledávání stavového prostoru hry na zadaný počet tahů dopředu,
 minimaxový algoritmus založený na statickém hodnocení pozic překračujících hloubku prohledávání."
 
 Classic 5-in-a-row game using Minimax algorithm
 
IDEAS:
 - alphabeta pruning
 - caching
 - GUI
 - non-square size
 - changing size after each round
 - finish print_grid (better characters to mark last move)
 - checking the result of the algorithm
 - split into more functions/files
 - keep track of wins/losses/draws
 - difficulty selection
 - change the order of methods
 - input 1-size and not 0-(size-1)
 - tile numbers around board
 - more classes (would it be even useful?)
 - safer input methods (against 'mean' user)
*/
#include <iostream>
class Game // main class that controls the whole program
{
public:
    Game();
    void game_loop(); // loops rounds until user decides to quit the program
    void two_computers(); // computer vs computer mode
private:
    int size; // board size, square shape
    int max_depth = 4; // max depth of minimax recursion, increases difficulty (but also time complexity exponentially)
    // (3..fast, 4..still okay, 5..really slow, 6+..ultra slow/eternity)
    int moves_left; // to track if there is still space left on the board
    enum tile {empty, human, computer, neighbour};
    enum winner {noone, user, algorithm, draw};
    enum direction {horizontal, vertical, diag_lu_rd, diag_ld_ru}; // left/right, up/down; better names?
    winner current_winner;
    tile ** grid;
    tile ** grid_copy; // against data corruption
    int result_row, result_col, player_row, player_col; // result of the minimax, players last move
    int vectors [8][2] = {{-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}}; // tiles with distance 1 from origin
    int vectors2 [16][2] = {{-2, 0}, {-2, 1}, {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2}, {2, 1}, {2, 0}, {2, -1}, {2, -2}, {1, -2}, {0, -2}, {-1, -2}, {-2, -2}, {-2, -1}}; // tiles with distance 2 from origin
    int values_table [3][6] = {{0, 0, 4, 16, 64, 1000000}, {0, 0, 2, 8, 32, 1000000}, {0, 0, 1, 4, 16, 1000000}}; // for static evaluation
    void prepare_grid(); // prepares board before every round
    void print_grid(int last_change_r, int last_change_c); // console output of the current board state
    bool game_over(tile player); // decides if there are any 5-a-row in current board state
    bool game_over_direction(int p1, int p2, int v1, int v2, direction mode, tile player); // general function for all possible directions
    int static_evaluation(tile player); // evaluates a position and how favorable it is for the computer
    int evaluation_direction(int p1, int p2, int v1, int v2, direction mode, tile player); // general function for all possible directions
    void find_neighbours(bool distance1, bool distance2); // finds adjacent tiles to the (algorithm's) starting position
    void insert_in_copy();
    int minimax(int depth, bool max); // branches off leaves, gameovers and nodes
    int minimax_compare(int depth, bool max, int value, tile player); // finds new tiles to deepen the algorithm into
    int minimax_core(int depth, bool max, int value, int i, int j); // finds the optimal move and stores it
    void player_move(); // player's input
    void turn_loop(); // loops turns until the round results with a win or a draw
    void print_grid_copy(); // debugging
    void print_vectors(); // debugging
    void insert_premade(); // debugging
    void two_computers_turn_loop(); // turn loop for computer vs computer
};
Game::Game()
{
    // game setting:
    char c;
    std::cout << "Play player vs computer (n) or computer vs computer (s)?\n";
    std::cin >> c;
    while(!(c == 'n' || c == 's'))
    {
        std::cout << "Incorrect input, try again:\n";
        std::cin >> c;
    }
    
    // set board size:
    std::cout << "Enter board size (NxN, enter N: 5-32):\n";
    std::cin >> this->size;
    while(!(this->size > 4 && this->size < 33))
    {
        std::cout << "Incorrect input, try again:\n";
        std::cin >> this->size;
    }
    
    // create board:
    this->grid = new tile*[this->size]; // row
    for(int i = 0; i < this->size; ++i)
        this->grid[i] = new tile[this->size]; // column
    
    this->grid_copy = new tile*[this->size]; // row
    for(int i = 0; i < this->size; ++i)
        this->grid_copy[i] = new tile[this->size]; // column
    
    this->prepare_grid();
    
    // start up game:
    if(c == 'n')
    {
        this->game_loop();
    }
    else
    {
        this->two_computers();
    }
}
void Game::game_loop() // loops rounds until user decides to quit the program
{
    while(1)
    {
        std::cout << "New round started:\n";
        
        // select starting player:
        std::cout << "Select starting player (c...computer, y...you):\n";
        char c;
        std::cin >> c;
        while(c != 'c' && c != 'y')
        {
            std::cout << "Incorrect input, try again please:\n";
            std::cin >> c;
        }
        if(c == 'c')
        {
            int rand_r, rand_c;
            rand_r = rand() % (this->size - 2) + 1; // borders are not that good of a starting move
            rand_c = rand() % (this->size - 2) + 1;
            this->print_grid(-1, -1);
            std::cout << "Computer starts with the move: " << rand_r << " " << rand_c << "\n";
            this->grid[rand_r][rand_c] = computer;
            this->moves_left--;
        }
        
        this->turn_loop();
        
        // round ended:
        if(this->current_winner == user)
        {
            std::cout << "You have won !!!\n";
        }
        else if (this->current_winner == algorithm)
        {
            std::cout << "Computer has won with a move " << this->result_row << " " << this->result_col << " !!!\n";
            this->grid[this->result_row][this->result_col] = computer;
            this->print_grid(-1, -1);
        }
        else // ~draw (use elif?)
        {
            this->print_grid(-1, -1);
            std::cout << "Tie, no moves left !!!\n";
        }
        
        // start new round:
        std::cout << "Do you wanna play another round? (y...yes, n...no)\n";
        std::cin >> c;
        while(c != 'y' && c != 'n')
        {
            std::cout << "Incorrect input, try again please:\n";
            std::cin >> c;
        }
        if(c == 'n')
        {
            break;
        }
        this->prepare_grid();
        this->current_winner = noone;
    }
}
void Game::turn_loop()  // loops turns until the round results with a win or a draw
{
    while(1)
    {
        this->print_grid(-1, -1);
        
        // player:
        this->player_move();
        if(--(this->moves_left) == 0)
        {
            this->current_winner = draw;
            break;
        }
        this->print_grid(-1, -1);
        
        // computer:
        std::cout << "Computer is thinking...\n";
        this->insert_in_copy();
        this->find_neighbours(true, false);
        this->minimax(0, true);
        if(this->current_winner != noone)
        {
            break;
        }
        std::cout << "Computer thinks " << this->result_row << " " << this->result_col << " is a good move:\n";
        this->grid[this->result_row][this->result_col] = computer;
        if(--(this->moves_left) == 0)
        {
            this->current_winner = draw;
            break;
        }
    }
}
void Game::prepare_grid() // prepares board before every round
{
    // clear grid tiles, winner, result, moves left:
    for(int i=0; i < this->size; i++)
    {
        for(int j=0; j < this->size; j++)
        {
            this->grid[i][j] = empty;
        }
    }
    this->current_winner = noone;
    this->result_row = -1;
    this->result_col = -1;
    this->moves_left = this->size * this->size;
}
void Game::print_grid(int last_change_r, int last_change_c) // console output of the current board state
{
    // tracks last change to be more easily noticable on larger boards (to-do)
    for(int i=0; i < this->size; i++)
    {
        for(int j=0; j < this->size; j++)
        {
            if(i == last_change_r && j == last_change_c)
            {
                std::cout << "#";
            }
            else
            {
                if(this->grid[i][j] == empty)
                {
                    std::cout << "_";
                }
                else if(this->grid[i][j] == human)
                {
                    std::cout << "X";
                }
                else if(this->grid[i][j] == computer)
                {
                    std::cout << "O";
                }
            }
            
            if(j != this->size - 1)
            {
                std::cout << " ";
            }
        }
        std::cout << "\n"; // endl better ?
    }
}
bool Game::game_over(tile player) // decides if there are any 5-a-row in current board state (for a certain player)
{
    // 4 possible directions (diagonals consist of upper and lower halves)
    if(this->game_over_direction(0, 0, 0, 1, horizontal, player))
    {
        return true;
    }
    if(this->game_over_direction(0, 0, 0, 1, vertical, player))
    {
        return true;
    }
    if(this->game_over_direction(0, 0, -1, 1, diag_ld_ru, player))
    {
        return true;
    }
    if(this->game_over_direction(1, this->size - 1, 1, -1, diag_ld_ru, player))
    {
        return true;
    }
    if(this->game_over_direction(0, 0, -1, 1, diag_lu_rd, player))
    {
        return true;
    }
    if(this->game_over_direction(1, this->size - 1, 1, -1, diag_lu_rd, player))
    {
        return true;
    }
    return false;
}
bool Game::game_over_direction(int p1, int p2, int v1, int v2, direction mode, tile player) // general function for all possible directions
{
    // p1, p2 parameters, v1, v2 for increments
    int streak = 0; // streak of player's tiles in a row
    tile current_tile;
    for(int d = p1; d < this->size; d++)
    {
        for(int i = d, j = p2; i > -1 && i < this->size && j < this->size; i = i + v1, j = j + v2)
        {
            switch (mode)
            {
                case horizontal:
                    current_tile = this->grid_copy[d][j];
                    break;
                case vertical:
                    current_tile = this->grid_copy[j][d];
                    break;
                case diag_lu_rd:
                    current_tile = this->grid_copy[i][j];
                    break;
                case diag_ld_ru:
                    current_tile = this->grid_copy[j][this->size - 1 - i];
                    break;
            }
            if(current_tile == player)
            {
                if(++streak == 5)
                {
                    return true;
                }
            }
            else
            {
                streak = 0;
            }
        }
        streak = 0;
    }
    return false;
}
int Game::static_evaluation(tile player) // evaluates a position and how favorable it is for the computer
{
    // very similar to the game-over method
    int value = 0; // ~the higher the better for the player
    // 4 possible directions (diagonals consist of upper and lower halves)
    value += this->evaluation_direction(0, 0, 0, 1, horizontal, player);
    value += this->evaluation_direction(0, 0, 0, 1, vertical, player);
    value += this->evaluation_direction(0, 0, -1, 1, diag_ld_ru, player);
    value += this->evaluation_direction(1, this->size - 1, 1, -1, diag_ld_ru, player);
    value += this->evaluation_direction(0, 0, -1, 1, diag_lu_rd, player);
    value += this->evaluation_direction(1, this->size - 1, 1, -1, diag_lu_rd, player);
    return value;
}
int Game::evaluation_direction(int p1, int p2, int v1, int v2, direction mode, tile player)
{
    // p1, p2 parameters, v1, v2 for increments
    int value = 0;
    int streak = 0; // streak of player's tiles in a row
    // tile streaks that are not blocked at the start or end are better
    // => values_table has values for streaks not blocked from either side ([0]), from one side ([1]) and from both sides ([2])
    int blocked_back = 1;
    tile current_tile;
    for(int d = p1; d < this->size; d++)
    {
        for(int i = d, j = p2; i > -1 && i < this->size && j < this->size; i = i + v1, j = j + v2)
        {
            switch (mode)
            {
                case horizontal:
                    current_tile = this->grid_copy[d][j];
                    break;
                case vertical:
                    current_tile = this->grid_copy[j][d];
                    break;
                case diag_lu_rd:
                    current_tile = this->grid_copy[i][j];
                    break;
                case diag_ld_ru:
                    current_tile = this->grid_copy[j][this->size - 1 - i];
                    break;
            }
            
            if(current_tile == player)
            {
                streak++;
            }
            else if(current_tile == empty || current_tile == neighbour)
            {
                value += this->values_table[blocked_back][streak];
                streak = 0;
                blocked_back = 0;
            }
            else
            {
                value += this->values_table[blocked_back + 1][streak];
                streak = 0;
                blocked_back = 1;
            }
        }
        if(streak != 0)
        {
            value += this->values_table[blocked_back + 1][streak];
            streak = 0;
        }
        blocked_back = 1;
    }
    return value;
}
void Game::find_neighbours(bool distance1, bool distance2) // finds adjacent tiles to the (algorithm's) starting position
{
    // instead of trying all empty tiles on the board we choose just the 'closer' ones:
    for(int i=0; i < this->size; i++)
    {
        for(int j=0; j < this->size; j++)
        {
            if(this->grid_copy[i][j] == human || this->grid_copy[i][j] == computer)
            {
                // viable origin tile found (i, j)
                if(distance1)
                {
                    for(int k=0; k<8; k++) // testing all 8 adjacent tiles
                    {
                        if(i + this->vectors[k][0] > -1 && i + this->vectors[k][0] < this->size && j + this->vectors[k][1] > -1 && j + this->vectors[k][1] < this->size)
                        {
                            if(this->grid_copy[i + this->vectors[k][0]][j + this->vectors[k][1]] == empty)
                            {
                                this->grid_copy[i + this->vectors[k][0]][j + this->vectors[k][1]] = neighbour;
                            }
                        }
                    }
                }
                
                if(distance2)
                {
                    for(int k=0; k<16; k++) // testing all 16 tiles (2 tiles away from origin)
                    {
                        if(i + this->vectors2[k][0] > -1 && i + this->vectors2[k][0] < this->size && j + this->vectors2[k][1] > -1 && j + this->vectors2[k][1] < this->size)
                        {
                            if(this->grid_copy[i + this->vectors2[k][0]][j + this->vectors2[k][1]] == empty)
                            {
                                this->grid_copy[i + this->vectors2[k][0]][j + this->vectors2[k][1]] = neighbour;
                            }
                        }
                    }
                }
            }
        }
    }
}
void Game::insert_in_copy()
{
    // copies into grid_copy:
    for(int i=0; i < this->size; i++)
    {
        for(int j=0; j < this->size; j++)
        {
            this->grid_copy[i][j] = this->grid[i][j];
        }
    }
}
int Game::minimax(int depth, bool max) // branches off leaves, gameovers and nodes
{
    // max depth reached (or no space on the board left):
    if(depth == std::min(this->max_depth, this->moves_left))
    {
        return (this->static_evaluation(computer) - this->static_evaluation(human));
    }
    
    // leaf ends with a win for someone:
    if(this->game_over(human))
    {
        if((max && depth == 0) || (!max && depth == 1))
        {
            this->current_winner = user;
            return -1000000;
        }
        return (this->static_evaluation(computer) - this->static_evaluation(human));
    }
    else if (this->game_over(computer))
    {
        if((!max && depth == 1) || (max && depth == 0))
        {
            this->current_winner = algorithm;
            return 1000000;
        }
        return (this->static_evaluation(computer) - this->static_evaluation(human));
    }
    
    // branching:
    if(max) // ~maximizing player ~computer
    {
        return minimax_compare(depth, max, -1000000, computer);
    }
    else // ~user
    {
        return minimax_compare(depth, max, 1000000, human);
    }
}
int Game::minimax_compare(int depth, bool max, int value, tile player) // finds new tiles to deepen the algorithm into
{
    int newly_added = 0; // tracks temporal_neighbours
    int temporal_neighbours [8][2]; // new viable tiles (neighbours) are added dynamically
    for(int i=0; i < this->size; i++)
    {
        for(int j=0; j < this->size; j++)
        {
            if(this->grid_copy[i][j] == neighbour) // possible tile found
            {
                this->grid_copy[i][j] = player; // insert players tile in this node
                
                // find new viable tiles:
                for(int k=0; k<8; k++)
                {
                    if(i + this->vectors[k][0] > -1 && i + this->vectors[k][0] < this->size && j + this->vectors[k][1] > -1 && j + this->vectors[k][1] < this->size)
                    {
                        if(this->grid_copy[i + this->vectors[k][0]][j + this->vectors[k][1]] == empty)
                        {
                            this->grid_copy[i + this->vectors[k][0]][j + this->vectors[k][1]] = neighbour;
                            temporal_neighbours[newly_added][0] = i + this->vectors[k][0];
                            temporal_neighbours[newly_added][1] = j + this->vectors[k][1];
                            newly_added++;
                        }
                    }
                }
                
                value = minimax_core(depth, max, value, i, j);
                
                if(this->current_winner != noone)
                {
                    return 1000000;
                }
                
                // delete dynamically added neighbours:
                for(int k=0; k < newly_added; k++)
                {
                    this->grid_copy[temporal_neighbours[k][0]][temporal_neighbours[k][1]] = empty;
                }
                
                newly_added = 0;
                this->grid_copy[i][j] = neighbour; // reset back to original
            }
        }
    }
    return  value;
}
int Game::minimax_core(int depth, bool max, int value, int i, int j) // finds the optimal move and stores it
{
    if(max)
    {
        if(depth == 0) // ~we are looking for best move in this depth (directly next move)
        {
            int value_before = value;
            value = std::max(value, minimax(depth + 1, false)); // loops deeper into the state space
            if(value > value_before || this->current_winner != noone)
            {
                this->result_row = i;
                this->result_col = j;
            }
        }
        else
        {
            value = std::max(value, minimax(depth + 1, false)); // loops deeper into the state space
        }
    }
    else
    {
        if(depth == 0) // ~we are looking for best move in this depth (directly next move)
        {
            int value_before = value;
            value = std::min(value, minimax(depth + 1, true)); // loops deeper into the state space
            if(value < value_before || this->current_winner != noone)
            {
                this->result_row = i;
                this->result_col = j;
            }
        }
        else
        {
            value = std::min(value, minimax(depth + 1, true)); // loops deeper into the state space
        }
    }
    return value;
}
void Game::player_move() // player's input
{
    // player's input:
    int row = -1, column = -1;
    char c = 'n';
    while(c != 'y')
    {
        std::cout << "Enter Your move ('i j' : i - row 0-" << this->size - 1 << ", j - column 0-" << this->size - 1 << "):\n";
        std::cin >> row >> column;
        while(!(row > -1 && row < this->size && column > -1 && column < this->size && this->grid[row][column] == empty))
        {
            std::cout << "Incorrect input, try again:\n";
            std::cin >> row >> column;
        }
        std::cout << "Are you sure you wanna enter row: " << row << " and column: " << column << " (y/n) ?\n";
        this->print_grid(row, column);
        std::cin >> c;
    }
    
    // set the input:
    this->player_row = row;
    this->player_col = column;
    this->grid[row][column] = human;
}
void Game::print_grid_copy() // for debugging, prints out neighbours/non-neighbours
{
    for(int i=0; i < this->size; i++)
    {
        for(int j=0; j < this->size; j++)
        {
            std::cout << this->grid_copy[i][j] << " ";
        }
        std::cout << "\n";
    }
}
void Game::print_vectors() // for debugging
{
    for(int i=0; i<8; i++)
    {
        for(int j=0; j<2; j++)
        {
            std::cout << this->vectors[i][j] << " ";
        }
        std::cout << "\n";
    }
}
void Game::insert_premade() // for debugging
{
    /*
     // size8+
    this->grid[1][1] = human;
    this->grid[2][2] = computer;
    this->grid[1][2] = human;
    this->grid[1][3] = computer;
    this->grid[0][4] = human;
    this->grid[2][3] = computer;
    this->grid[3][3] = human;
    this->grid[2][4] = computer;
    this->grid[7][7] = human;
    this->grid[2][1] = computer;
     */
}
void Game::two_computers() // mode for computer vs computer
{
    srand (time(NULL));
    while(1)
    {
        std::cout << "New round started (computer vs computer):\n";
        
        int rand_r, rand_c;
        rand_r = std::rand() % (this->size - 2) + 1; // borders are not that good of a starting move
        rand_c = std::rand() % (this->size - 2) + 1;
        this->print_grid(-1, -1);
        std::cout << "Computer(1) starts with the move: " << rand_r << " " << rand_c << "\n";
        this->grid[rand_r][rand_c] = computer;
        this->moves_left--;
        
        this->two_computers_turn_loop();
        
        // round ended:
        if(this->current_winner == user)
        {
            std::cout << "Computer(2) has won with a move " << this->result_row << " " << this->result_col << " !!!\n";
            this->grid[this->result_row][this->result_col] = human;
            this->print_grid(-1, -1);
        }
        else if (this->current_winner == algorithm)
        {
            std::cout << "Computer(1) has won with a move " << this->result_row << " " << this->result_col << " !!!\n";
            this->grid[this->result_row][this->result_col] = computer;
            this->print_grid(-1, -1);
        }
        else // ~draw (use elif?)
        {
            this->print_grid(-1, -1);
            std::cout << "Tie, no moves left !!!\n";
        }
        
        // start new round:
        char c;
        std::cout << "Do you wanna see another round? (y...yes, n...no)\n";
        std::cin >> c;
        while(c != 'y' && c != 'n')
        {
            std::cout << "Incorrect input, try again please:\n";
            std::cin >> c;
        }
        if(c == 'n')
        {
            break;
        }
        this->prepare_grid();
        this->current_winner = noone;
    }
}
void Game::two_computers_turn_loop()  // computer vs computer: loops turns until the round results with a win or a draw
{
    while(1)
    {
        this->print_grid(-1, -1);
        
        // computer(2):
        std::cout << "Computer(2) is thinking...\n";
        this->insert_in_copy();
        this->find_neighbours(true, false);
        this->minimax(0, false);
        if(this->current_winner != noone)
        {
            break;
        }
        std::cout << "Computer(2) thinks " << this->result_row << " " << this->result_col << " is a good move:\n";
        this->grid[this->result_row][this->result_col] = human;
        if(--(this->moves_left) == 0)
        {
            this->current_winner = draw;
            break;
        }
        this->print_grid(-1, -1);
        
        // computer(1):
        std::cout << "Computer(1) is thinking...\n";
        this->insert_in_copy();
        this->find_neighbours(true, false);
        this->minimax(0, true);
        if(this->current_winner != noone)
        {
            break;
        }
        std::cout << "Computer(1) thinks " << this->result_row << " " << this->result_col << " is a good move:\n";
        this->grid[this->result_row][this->result_col] = computer;
        if(--(this->moves_left) == 0)
        {
            this->current_winner = draw;
            break;
        }
    }
}
int main(int argc, const char * argv[]) {
    std::cout << "Welcome to 5-in-a-row !\n";
    
    
    Game * my_game = new Game;
    delete my_game;
    
    return 0;
}
