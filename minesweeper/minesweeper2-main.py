import random as rnd
import sys
import itertools as it
import time
import tkinter as tk
from tkinter import messagebox

# version 1.2 of the 'Fair Minesweeper' game (made by: Josef Sykora):

# Topic description:
# Férové miny:
# Implementace klasického minesweeperu se dvěma vlastnostmi navíc: Pokud uživatel, soudě dle stavu hrací plochy,
# nemohl mít ani u jednoho políčka jistotu, že na něm mina není (např. vždy na začátku hry), mina po odkrytí
# libovolného políčka na tomto políčku není. Naopak pokud uživatel mohl mít jistotu, že na některém políčku mina není,
# a přesto odkryl políčko, u nějž tuto jistotu mít nemohl, mina na něm je.

# todo:
#  - overall visual with tkinter
#  - rewriting the tkinter functions - split existing ones, use some classes
#  - implement mine possibility for tiles that aren't just 0% and 100% (some guesses ARE actually better than others)
#  - give functions a better order
#  - add option of a custom size of board (and mines)
#  - add a 'density' of mines option (to the custom sized board)
#  - add a 'zoom' option in tkinter
#  - better win/screen
#  - add a 'highscore' feature (maybe even with a small database)
#  - add a progress bar of the program (what is it currently doing)
#  - add a 'live bot solving' feature
#  - add a 'terminate brute_force' functionality (so the program would'nt freeze in edge cases)
#  - add a 'ignore low-impact blocks' feature (a number has only one number neighbour etc.)
#  - add a 'first click on board is a zero' feature
#  - implement pre-counting of possible player moves (it would speed up the next move)
#  - look into solving partially base-2 equalities
#  - use the OOP more
#  - add the 'reset game' button (and equivalent of the smiley face - changes on click/release)


# class for the current game's board, contains all the information about, and the information we will need for
# some of the functions
class Grid:
    def __init__(self, height, width, mines_number):
        self.height = height
        self.width = width
        self.mines_number = mines_number  # starting number of mines, doesnt change
        self.grid = self.setup_grid()
        self.active = True  # shuts down board's functionality after losing/winning
        self.pending = []  # keeps track of revealed numbers on board that are yet to be solved
        self.chunks = []  # keeps track of separate chunks of tiles that don't affect each other
        self.independent = []  # hidden tiles that have currently no adjacent (revealed) numbers
        self.found_mines = 0
        self.uncovered = 0

    # creates the grid itself
    def setup_grid(self):
        grid = self.plant_mines()
        grid = self.connect_tiles(grid)
        grid = self.setup_numbers(grid)
        return grid

    # randomly places starting mines on the board
    def plant_mines(self):
        grid = []

        shuffle_deck = ([True] * self.mines_number) + ([False] * ((self.height * self.width) - self.mines_number))
        rnd.shuffle(shuffle_deck)

        for i in range(self.height):
            grid.append([])
            for j in range(self.width):
                if shuffle_deck[i * self.width + j]:
                    grid[i].append(Tile(-1, i, j))
                else:
                    grid[i].append(Tile(0, i, j))

        return grid

    # connects all adjacent tiles to each other
    def connect_tiles(self, grid):
        vectors = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

        for i in range(self.height):
            for j in range(self.width):
                for u, v in vectors:
                    if 0 <= i + u <= self.height - 1 and 0 <= j + v <= self.width - 1:
                        grid[i][j].adjacent.append(grid[i + u][j + v])

        return grid

    # calculates and places the numbers tiles on the board
    def setup_numbers(self, grid):
        for i in range(self.height):
            for j in range(self.width):
                if grid[i][j].number == -1:
                    # increases all tiles around a mine
                    for tile in grid[i][j].adjacent:
                        if tile.number != -1:
                            tile.number += 1

        return grid

    # debugging: prints out the back of the grid (even the hidden parts)
    def print_back(self):
        print('back:')
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if self.grid[i][j].number == -1:
                    row.append('#')
                else:
                    row.append(str(self.grid[i][j].number))
            print(row)

    # debugging: prints the front of the grid (what the player sees)
    def print_front(self):  # for testing purposes
        print('front:')
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if self.grid[i][j].hidden:
                    if self.grid[i][j].info == 'no':
                        row.append('_')
                    if self.grid[i][j].info == 'safe':
                        row.append('+')
                    if self.grid[i][j].info == 'danger':
                        row.append('#')
                else:
                    row.append(str(self.grid[i][j].number))
            print(row)


# class representing a single tile from the game's board, contains it's information
class Tile:
    def __init__(self, number, row, col):
        self.hidden = True
        self.number = number  # 0-9 for number, -1 for mine
        self.info = 'no'  # contains program's information about the tile: 'no'/'safe'/'danger'
        self.pending = False
        self.adjacent = []
        self.row = row
        self.col = col  # (column)
        self.chunk = None  # in which chunk is the tile currently
        self.found_in = ''  # in which function did the program solve this tile: 'obvious'/'brute'/'count'


# class to contain groups of tiles (both numbers and hidden tiles) that affect each other (~ they are adjacent)
class Chunk:
    def __init__(self, hidden, numbers):
        self.hidden = hidden
        self.numbers = numbers
        self.mines_min = None  # todo: delete
        self.possible_mines = []  # keeps tracks of all possible amounts of mines in the chunk
        self.mines_max = 0  # todo: delete
        self.mines_placement = None  # container used in functions using brute force
        self.correct_combinations = 0
        self.all_combinations = []  # contains Combination instances


# class used to store all possible combinations tied to a chunk
# note: could be possibly done with tuples instead of class, class seems more readable though
class Combination:
    def __init__(self):
        self.values = []  # True ~ mine, False ~ no mine
        self.mines_count = 0  # number of mines in a single Combination instance


# debugging: converts a group of Tile instances into more easily printable list
def chunk_print(chunk):
    array = []
    for tile in chunk:
        array.append((tile.row, tile.col))
    return array


# debugging: checks all chunks for: their length vs. their combinations
def check_chunks(grid):
    for chunk in grid.chunks:
        if chunk.all_combinations:
            if len(chunk.hidden) != len(chunk.all_combinations[0].values):
                print('combination too long for this chunk')
                sys.exit()
    return grid


# debugging: prints every tile's .chunk
def print_chunks_grid(grid):  # for testing, delete later
    for i in range(grid.height):
        line = []
        for j in range(grid.width):
            if grid.grid[i][j].chunk is None:
                line.append('__main__.Chunk object at 0x000000000000')
            else:
                line.append(grid.grid[i][j].chunk)
        print(line)


# debugging: checks number of mines on the board and if all the numbers on board are correct (after reload_grid())
def check_board(grid):
    mines_cur = 0
    incorrect = []
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j].number == -1:
                mines_cur += 1
            else:
                mines_around = 0
                for tile in grid.grid[i][j].adjacent:
                    if tile.number == -1:
                        mines_around += 1
                if mines_around != grid.grid[i][j].number:
                    incorrect.append((i, j))
    if incorrect or mines_cur != grid.mines_number:
        print('board is set up incorrectly:')
        print(mines_cur, incorrect)
        sys.exit()


# processes the player's input, triggers the 'Fair minesweeper' functionality with function reload_grid() if needed
def player_move(grid, current_tile):
    safes_in_grid = safes_existence(grid)  # checks if there are any 'safe' tiles on the board
    if safes_in_grid:
        if current_tile.info == 'no':
            if current_tile.number != -1:  # kills the player (he made a wrong move)
                grid = reload_grid(grid, current_tile, -1)

    else:
        if current_tile.info == 'no':
            if current_tile.number == -1:  # saves the player (he could'nt make a better move)
                grid = reload_grid(grid, current_tile, 0)

    grid = uncover_tile(grid, current_tile)
    return grid


# reloads the grid with a random (but correct) layout that has or doesnt have a mine at the desired location
# similar to solve_mine_count()
# todo: split into more functions
def reload_grid(grid, current_tile, insert):
    grid.independent = []
    grid = find_independent(grid)  # updates the independent list
    unchangeable_sum = 0  # chunks that have fixed amount of mines cannot change in this algorithm
    changeable_chunks = []  # chunks that don't have fixed amount of mines

    for chunk in grid.chunks:  # sorts all chunks into the 2 groups
        if len(chunk.possible_mines) == 1:
            unchangeable_sum += chunk.possible_mines[0]
        else:
            changeable_chunks.append(chunk)
            # resets the number tiles:
            grid = clean_group(grid, chunk.hidden)
            grid = clean_group(grid, chunk.numbers)

    # resets the rest:
    if current_tile.chunk is not None:
        grid = clean_group(grid, current_tile.chunk.numbers)
    grid = clean_group(grid, grid.independent)
    grid = clean_all(grid)

    # replanting the mines:

    mines_to_replant = grid.mines_number - grid.found_mines - unchangeable_sum
    if current_tile.chunk is None:
        current_tile.number = insert
        grid.independent.remove(current_tile)
        if insert == -1:
            mines_to_replant -= 1

    else:  # we need to replant the tile's original chunk:
        grid, placed_mines = handle_origin(grid, current_tile, insert)
        if current_tile.chunk in changeable_chunks:
            mines_to_replant -= placed_mines
            changeable_chunks.remove(current_tile.chunk)

    if grid.independent or changeable_chunks:
        grid, _, numbers_option = find_wrong_numbers(grid, changeable_chunks, mines_to_replant)
        chosen_numbers = rnd.choice(numbers_option)  # chooses a random correct number of mines for each chunk
        for i in range(len(chosen_numbers) - 1):
            # picks a random combination in every chunk and plants the mines accordingly:
            chosen_combination = choose_combination(changeable_chunks[i], chosen_numbers[i])
            for j in range(len(chosen_combination.values)):
                if chosen_combination.values[j]:
                    changeable_chunks[i].hidden[j].number = -1
                else:
                    changeable_chunks[i].hidden[j].number = 0

        # plants mines in independent:
        shuffle_deck = ([True] * chosen_numbers[-1]) + ([False] * (len(grid.independent) - chosen_numbers[-1]))
        rnd.shuffle(shuffle_deck)
        for i in range(len(grid.independent)):
            if shuffle_deck[i]:
                grid.independent[i].number = -1

    grid = set_all_numbers(grid)  # sets all the numbers back up
    return grid


# sets all numbers in grid after the reload
def set_all_numbers(grid):
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j].number == -1:  # increase all numbers around the mine
                for tile in grid.grid[i][j].adjacent:
                    if tile.number != -1:
                        tile.number += 1

    return grid


# chooses a random combination (that satisfies the mines amount that is needed) of mines for a single chunk
def choose_combination(chunk, number):
    viable_combinations = []
    for combination in chunk.all_combinations:
        if combination.mines_count == number:
            viable_combinations.append(combination)

    chosen_combination = rnd.choice(viable_combinations)
    return chosen_combination


# chooses a random combination for the chunk that has/doesnt have a mine at a specific place
def handle_origin(grid, current_tile, insert):
    origin = current_tile.chunk
    placed_mines = 0  # needs to be reduced from the number of mines to replant later
    viable_combinations = []
    index = origin.hidden.index(current_tile)

    for combination in origin.all_combinations:
        if (insert == -1 and combination.values[index]) or (insert == 0 and not combination.values[index]):
            viable_combinations.append(combination)

    chosen_combination = rnd.choice(viable_combinations)
    for i in range(len(origin.hidden)):  # makes changes in the chunk
        if chosen_combination.values[i]:
            origin.hidden[i].number = -1
            placed_mines += 1
        else:
            origin.hidden[i].number = 0

    return grid, placed_mines


# cleans all the numbers in a certain group of tiles
def clean_group(grid, chunk):
    for tile in chunk:
        tile.number = 0
    return grid


# cleans the whole board of the old numbers
def clean_all(grid):
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j].number != -1:
                grid.grid[i][j].number = 0
    return grid


# searches the whole board for any tiles marked with 'safe'; True ~ found at least one, False ~ found none
def safes_existence(grid):
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j].info == 'safe' and grid.grid[i][j].hidden:
                return True

    return False


# solves by trying all possible (correct) combinations in every chunk
def solve_brute_force(grid):
    grid, all_chunks = find_chunks(grid)  # finds all chunks on the current board
    new_chunks = select_new(grid, all_chunks)  # selects new chunks from them
    changed_chunks = []

    for chunk in new_chunks:
        # finds all the combinations and makes changes
        grid, changed = handle_chunk(grid, chunk, 'brute_force', [])
        if changed == len(chunk.hidden):
            all_chunks.remove(chunk)
        elif 0 < changed < len(chunk.hidden):
            all_chunks.remove(chunk)
            changed_chunks.append(chunk)

    grid = update_newly_found(grid, changed_chunks, all_chunks)  # finds new chunks and finds new combinations in them
    grid.chunks = all_chunks
    return grid


visited = []  # used for tracking progress in the recursive function visit_tile()
chunk_to_add = Chunk([], [])  # new chunk created in the find_chunks()


# finds all chunks in the current board
def find_chunks(grid, pending=None):
    global visited
    global chunk_to_add
    if pending is None:
        pending = grid.pending
    visited = [[False] * grid.width for _ in range(grid.height)]
    all_chunks = []

    for tile in pending:  # every chunk must contain at least one uncovered number tile
        if not visited[tile.row][tile.col] and tile.pending:
            chunk_to_add = Chunk([], [])
            visit_tile(grid, tile, True)  # starts the recursive search from uncovered number tile

            chunk_to_add.mines_placement = [0] * len(chunk_to_add.hidden)
            chunk_to_add.mines_min = len(chunk_to_add.hidden)

            all_chunks.append(chunk_to_add)

    return grid, all_chunks


# recursive function that finds all tiles inside of a chunk (~ all tiles that are linked to the starting tile)
def visit_tile(grid, start_tile, parity):
    global visited
    global chunk_to_add

    if parity:  # parity to switch between numbers and blanks: True ~ number; False ~ hidden
        chunk_to_add.numbers.append(start_tile)
    else:
        chunk_to_add.hidden.append(start_tile)
    visited[start_tile.row][start_tile.col] = True

    for tile in start_tile.adjacent:  # continues the search in the adjacent tiles
        if not visited[tile.row][tile.col]:
            if parity:
                if tile.info == 'no':
                    visit_tile(grid, tile, not parity)
            else:
                if grid.grid[tile.row][tile.col].pending:
                    visit_tile(grid, tile, not parity)


# compares old chunks to the newly found and selects ones that are not repeated
def select_new(grid, all_chunks):
    new_chunks = []

    for chunk in all_chunks:  # iterates over all pairs of {1 old + 1 new} chunks
        new_hid, new_num = [], []  # rewrites new chunk as tuples (tile instances wouldn't be same in old and new)
        for item in chunk.hidden:
            new_hid.append((item.row, item.col))
        for item in chunk.numbers:
            new_num.append((item.row, item.col))
        is_new = True

        for old_chunk in grid.chunks:
            if len(chunk.hidden) == len(old_chunk.hidden) and len(chunk.numbers) == len(old_chunk.numbers):
                old_hid, old_num = [], []  # rewrites original chunk as tuples
                for tile in old_chunk.hidden:
                    old_hid.append((tile.row, tile.col))
                for tile in old_chunk.numbers:
                    old_num.append((tile.row, tile.col))

                if set(new_hid) == set(old_hid) and set(new_num) == set(old_num):  # compares the chunks
                    is_new = False
                    all_chunks[all_chunks.index(chunk)] = old_chunk  # passes all the information from the old chunk

        if is_new:  # changes the info in corresponding tiles
            new_chunks.append(chunk)
            for tile in chunk.hidden + chunk.numbers:  # todo: check if it is working
                tile.chunk = chunk

    return new_chunks


# finds all combinations of mines for a chunk, makes changes in the chunk according to it
# todo: change wrong_numbers into an optional argument
def handle_chunk(grid, chunk, how, wrong_numbers):
    # determine what function did call this one:
    if how == 'brute_force':
        test_save = grid.pending
        try_combination(grid, chunk, 0)
        grid.pending = test_save
    elif how == 'mine_count':
        alter_combinations(grid, chunk, wrong_numbers)

    changed = 0

    # make the changes in blanks:
    for i in range(len(chunk.mines_placement)):
        if chunk.mines_placement[i] == chunk.correct_combinations:  # mine in all combinations
            chunk.hidden[i].info = 'danger'
            grid.found_mines += 1
            changed += 1

        if chunk.mines_placement[i] == 0:  # safe in all combinations
            chunk.hidden[i].info = 'safe'
            changed += 1

        if chunk.mines_placement[i] == chunk.correct_combinations or chunk.mines_placement[i] == 0:  # sets found_in
            if how == 'brute_force':
                chunk.hidden[i].found_in = 'brute'
            elif how == 'mine_count':
                chunk.hidden[i].found_in = 'count'

    # make the changes in numbers:
    for tile in chunk.numbers:
        not_pending = True
        for item in tile.adjacent:
            if item.info == 'no':
                not_pending = False
        if not_pending:
            grid.pending.remove(tile)
            tile.pending = False

    return grid, changed


# recursive function finding all the combinations in a chunk
def try_combination(grid, chunk, depth):
    # tries putting a 'danger' in a blank, tries solving it with solve_obvious, then tries putting a 'danger' in
    # the next 'empty' blank, ..., until all blanks are filled, checks if the combination is valid, after that
    # steps up one recursion level (~ one decision) and tries putting a 'safe' there, ..., until fully
    # emerges from the recursion

    for i in range(depth, len(chunk.hidden)):
        if chunk.hidden[i].info == 'no':
            # try 'danger':
            chunk.hidden[i].info = 'danger'
            grid, changes = solve_obvious(grid, chunk.numbers)
            chunk.numbers = grid.pending
            try_combination(grid, chunk, i + 1)
            grid = revert_changes(grid, changes)  # reverts the changes that were made on this recursion depth

            # try 'safe':
            chunk.hidden[i].info = 'safe'
            grid, changes = solve_obvious(grid, chunk.numbers)
            chunk.numbers = grid.pending
            try_combination(grid, chunk,  i + 1)
            grid = revert_changes(grid, changes)  # reverts the changes that were made on this recursion depth

            chunk.hidden[i].info = 'no'
            return

    set_min_max(grid, chunk)  # checks combination for validity and connects it to the chunk


# checks combination (also creates the actual Combination instance here) and appends it to the chunk's all_combinations
# todo: rename, delete min and max functionality (not necessary anymore)
def set_min_max(grid, chunk):
    if not grid.pending:  # ~ combination is correct
        chunk.correct_combinations += 1
        mines_count = 0
        current_combination = Combination()

        for i in range(len(chunk.hidden)):  # sets the info into the combination
            if chunk.hidden[i].info == 'danger':
                chunk.mines_placement[i] += 1
                mines_count += 1
                current_combination.values.append(True)
                current_combination.mines_count += 1
            else:
                current_combination.values.append(False)

        if mines_count not in chunk.possible_mines:
            chunk.possible_mines.append(mines_count)
        chunk.all_combinations.append(current_combination)

        if mines_count < chunk.mines_min:
            chunk.mines_min = mines_count
        if mines_count > chunk.mines_max:
            chunk.mines_max = mines_count


# a shorter version of function handle_chunk() (but doesnt make any changes in the tiles)
def only_min_max(grid, chunk):
    test_save = grid.pending
    try_combination(grid, chunk, 0)
    grid.pending = test_save
    return grid


# reverts the changes that the solve_obvious() has made called from try_combination()
def revert_changes(grid, changes):
    for tile in changes[0]:  # revert changes in blanks
        grid.grid[tile[0]][tile[1]].info = 'no'
        grid.grid[tile[0]][tile[1]].found_in = ''  # not necessary (just to keep it a bit cleaner)

    for tile in changes[1]:  # revert changes in numbers
        grid.grid[tile[0]][tile[1]].pending = True
        grid.pending.append(grid.grid[tile[0]][tile[1]])

    return grid


# finds new chunks and finds new combinations in them
def update_newly_found(grid, changed_chunks, into):
    for chunk in changed_chunks:
        save_pending = grid.pending
        grid, newly_found = find_chunks(grid, chunk.numbers)
        grid.pending = save_pending

        for item in newly_found:
            grid = only_min_max(grid, item)  # shorter version since all deductible solves were already made
        into.extend(newly_found)

    return grid


# discards all old combinations that have an unsuitable amount of mines in them and prepares changes to be made
def alter_combinations(grid, chunk, wrong_numbers):
    chunk.correct_combinations = 0
    chunk.mines_placement = [0] * len(chunk.hidden)  # prepare to apply changes (in handle_chunk())
    new_combinations = []

    for combination in chunk.all_combinations:
        if combination.mines_count not in wrong_numbers:
            new_combinations.append(combination)
            chunk.correct_combinations += 1
            for i in range(len(combination.values)):
                if combination.values[i]:
                    chunk.mines_placement[i] += 1

    chunk.all_combinations = new_combinations
    return grid


# solves grid by using the number of mines left and brute force
# similar to reload_grid()
def solve_mine_count(grid):
    #  looks at the already existing combinations from solve_brute_force() and assigns each chunk a number representing
    #  how many mines are in the particular combination in the chunk, then checks if the all those numbers added up are
    #  even possible on the current grid (there can be not enough or too many mines), some chunks have a set number of
    #  mines and thus cannot change anyhow (~ unchangeable chunks), the independent tiles are considered in this method
    #  as well, after finding numbers (of mines in individual chunks) that are incorrect, they are removed, this can
    #  lead to some deductions on some particular tiles in the chunks or in the independent pile

    grid = find_independent(grid)  # updates the independent list
    unchangeable_sum = 0  # chunks that have fixed amount of mines cannot change in this algorithm
    changeable_chunks = []  # chunks that don't have fixed amount of mines

    for chunk in grid.chunks:  # sorts all chunks into the 2 groups
        if len(chunk.possible_mines) == 1:
            unchangeable_sum += chunk.possible_mines[0]
        else:
            changeable_chunks.append(chunk)

    result = grid.mines_number - grid.found_mines - unchangeable_sum
    grid, wrong_numbers, _ = find_wrong_numbers(grid, changeable_chunks, result)  # discards incorrect combinations
    grid = make_changes(grid, changeable_chunks, wrong_numbers)  # applies changes according to the new combinations
    grid = find_independent(grid)  # probably not necessary (just for cleanness)
    return grid


# discards incorrect combinations from chunks (now taking the number of mines left in consideration)
def find_wrong_numbers(grid, changeable_chunks, result):
    numbers = []  # each list in this list represents corresponding possible numbers of mines in each chunk
    for chunk in changeable_chunks:
        list_copy = []
        for i in range(len(chunk.possible_mines)):
            list_copy.append(chunk.possible_mines[i])
        numbers.append(list_copy)

    numbers_from_independent = []
    for i in range(len(grid.independent) + 1):
        numbers_from_independent.append(i)
    numbers.append(numbers_from_independent)  # takes the independent in consideration, last item in the list

    combinations = list(it.product(*numbers))  # produces all combinations of the chunk numbers
    correct_combinations = []

    for item in combinations:  # checks if the mines add up to the correct total
        combination_sum = 0
        for i in range(len(numbers)):
            combination_sum += item[i]
        if combination_sum == result:
            correct_combinations.append(item)

    for item in correct_combinations:
        for i in range(len(numbers)):
            if item[i] in numbers[i]:  # incorrect combinations STAY in the numbers list
                numbers[i].remove(item[i])

    return grid, numbers, correct_combinations  # todo: better return (so everything would get used)


# applies the changes in the newly reduced combinations in chunks
def make_changes(grid, changeable_chunks, numbers):
    changed_chunks = []
    for i in range(len(numbers) - 1):
        if numbers[i]:  # ~ there are some incorrect combinations in this chunk
            # applies the changes in the chunk if possible:
            grid, changed = handle_chunk(grid, changeable_chunks[i], 'mine_count', numbers[i])
            if changed == len(changeable_chunks[i].hidden):
                grid.chunks.remove(changeable_chunks[i])
            elif 0 < changed < len(changeable_chunks[i].hidden):
                grid.chunks.remove(changeable_chunks[i])
                changed_chunks.append(changeable_chunks[i])

    # finds new chunks and finds new combinations in them
    grid = update_newly_found(grid, changed_chunks, grid.chunks)

    if len(numbers[-1]) == len(grid.independent):  # changes all the independent tiles
        # conclusive only if all independent are mines or all independent are safes
        if 0 not in numbers[-1]:
            for tile in grid.independent:
                tile.info = 'safe'
                tile.found_in = 'count'

        elif len(grid.independent) not in numbers[-1]:
            for tile in grid.independent:
                tile.info = 'danger'
                tile.found_in = 'count'

    return grid


# finds independent tiles (they don't affect any other mines)
def find_independent(grid):
    grid.independent = []

    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j].info == 'no' and grid.grid[i][j].chunk is None:
                grid.independent.append(grid.grid[i][j])

    return grid


changed_blanks = []  # used to store changes from try_combination so they can be reversed
changed_numbers = []  # -//-


# function solving 'obvious cases' (sufficient amount of mines is already around/ exact number of blanks around)
def solve_obvious(grid, pending=None):
    global changed_blanks
    global changed_numbers
    not_permanent = False
    found_mines = grid.found_mines

    if pending is not None:  # triggered by being called from a function using brute force
        grid.pending = pending
        not_permanent = True
        changed_blanks = []
        changed_numbers = []

    while True:  # iterates while there is still change being made
        before = len(grid.pending)
        grid = find_obvious(grid)
        if len(grid.pending) == before:
            break

    if not_permanent:
        grid.found_mines = found_mines
        return grid, (changed_blanks, changed_numbers)
    return grid


# solves as many 'obvious cases' at a time as it can
def find_obvious(grid):
    global changed_blanks
    global changed_numbers
    new_pending = []

    while grid.pending:  # iterates over all pending numbers
        current_tile = grid.pending.pop()
        current_tile.pending = False
        blanks_around = []
        mines_around = 0

        for tile in current_tile.adjacent:  # scans the surroundings
            if tile.info == 'danger':
                mines_around += 1
            elif tile.info == 'no':
                blanks_around.append(tile)

        if current_tile.number - mines_around == len(blanks_around):  # case: exact number of blanks around
            changed_numbers.append((current_tile.row, current_tile.col))
            for tile in blanks_around:
                tile.info = 'danger'
                tile.found_in = 'obvious'
                changed_blanks.append((tile.row, tile.col))
                grid.found_mines += 1

        elif current_tile.number - mines_around == 0:  # case: sufficient amount of mines is already around
            changed_numbers.append((current_tile.row, current_tile.col))
            for tile in blanks_around:
                tile.info = 'safe'
                tile.found_in = 'obvious'
                changed_blanks.append((tile.row, tile.col))

        else:
            new_pending.append(current_tile)
            current_tile.pending = True

    grid.pending = new_pending  # returns pending tiles without the solved ones
    return grid


# core solve function containing the 3 solving algorithms
def solve(grid):
    # optional print-outs in the console:
    # print('before obvious:')
    # grid.print_front()

    grid = solve_obvious(grid)
    # print('after obvious:')
    # grid.print_front()

    grid = solve_brute_force(grid)
    grid = cleanse(grid)
    # print('after brute_force:')
    # grid.print_front()

    grid = solve_mine_count(grid)
    grid = cleanse(grid)
    # print('after mine_count ~ final:')
    # grid.print_front()

    return grid


# corrects the tiles that should no longer be associated with a certain chunk, corrects the chunks as well
# by removing the incorrect tiles from them
def cleanse(grid):  # todo: rename?
    for i in range(grid.height):
        for j in range(grid.width):
            if (grid.grid[i][j].info != 'no' and grid.grid[i][j].hidden)\
                    or (not grid.grid[i][j].pending and not grid.grid[i][j].hidden):
                grid.grid[i][j].chunk = None

    for chunk in grid.chunks:
        for tile in chunk.hidden:
            if tile.info != 'no':
                chunk.hidden.remove(tile)

        for tile in chunk.numbers:
            if not tile.pending:
                chunk.numbers.remove(tile)

    return grid


# todo udelej lepsi prebrani otcovskych __init__ (args, kwargs)
# todo koukni se na self.register
# todo predavani 'controller'
# todo nastavit lepsi minimum/maximum pro vysku/sirku (73x43 ?)


# shell for the entire game and all the tkinter widgets
class MainFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame
        self.parent = parent

        container = tk.Frame(self)  # container for switching in between main screens (shows one at a time)
        container.grid(row=0, column=0)

        self.screens = {}  # dictionary of the screen instances

        for S in (MenuFrame, GameFrame):  # sets up all the screens into the dictionary
            screen = S(container, self)
            self.screens[S] = screen

        self.screens[MenuFrame].grid(row=0, column=0)  # starting screen is the Menu

    # removes the current screen and loads desired screen
    def show_screen(self, new_screen, screen_to_remove):
        screen_to_remove.grid_remove()
        frame = self.screens[new_screen]
        frame.grid(row=0, column=0)


# screen option containing all the menu elements
class MenuFrame(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.title = tk.Label(self, text='Fair Minesweeper', font=('Helvetica', 30))
        self.title.grid(row=0, column=0)

        self.settings = SettingsFrame(self, self.controller)  # shows all the options the player can choose
        self.settings.grid(row=1, column=0)

        self.play = PlayButton(self, self.controller)  # button for starting the game
        self.play.grid(row=2, column=0)


# contains options selectable by player
class SettingsFrame(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.difficulty = DifficultyFrame(self, self.controller)  # operates all the height/width/mine options
        self.difficulty.grid(row=0, column=0)

        self.bot = BotCheckBox(self, self.controller)  # switch to show the solving bot during game
        self.bot.grid(row=1, column=0)

        self.mines = MinesCheckBox(self, self.controller)  # switch to show position of all the mines during game
        self.mines.grid(row=2, column=0)

    # extracts info from children widgets
    def get_info(self):
        # returns: (height, width, number of mines), solving bot turned on/off, seeing mines on/off
        return self.difficulty.get_info(), self.bot.get_info(), self.mines.get_info()


# shell of OptionsFrame : contains options for boards dimensions and number of mines inside
class DifficultyFrame(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.label = tk.Label(self, text='Choose difficulty:')
        self.label.grid(row=0, column=0)

        self.options = OptionsFrame(self, self.controller)
        self.options.grid(row=0, column=1)

    # extracts (height/width/mines)
    def get_info(self):
        return self.options.get_info()


# contains options for boards dimensions and number of mines inside
class OptionsFrame(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.chosen = tk.StringVar()  # keeps track of which difficulty is currently chosen
        self.chosen.set('Easy')  # default difficulty is Easy

        # contains difficulties with a set height/width/mines
        self.classic = ClassicDifficulties(self, self.controller, self.chosen)
        self.classic.grid(row=0, column=0)

        # contains difficulty where the player can set the height/width/mines manually
        self.custom = CustomDifficulty(self, self.controller, self.chosen)
        self.custom.grid(row=1, column=0)

    #  finds out whether the difficulty is 'classic' or 'custom' and gets the height/width/mines
    def get_info(self):
        if self.chosen.get() in self.classic.classics.keys():
            return self.classic.get_info()
        elif self.chosen.get() == 'Custom':
            return self.custom.get_info()


# contains 'classic' difficulties with a set height/width/mines
class ClassicDifficulties(tk.Frame):
    def __init__(self, parent, controller, chosen, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.chosen = chosen  # keeps track of which difficulty is selected

        # info with height/width/mines
        self.classics = {
            'Easy': {
                'place': 0,
                'height': 9,
                'width': 9,
                'mines': 10
            },
            'Medium': {
                'place': 1,
                'height': 16,
                'width': 16,
                'mines': 40
            },
            'Hard': {
                'place': 2,
                'height': 16,
                'width': 30,
                'mines': 99
            },
        }

        for key in self.classics:  # loads all options on screen
            radio_button = tk.Radiobutton(self, text=key, variable=self.chosen, value=key)
            radio_button.grid(row=self.classics[key]['place'], column=0)

    # one of the ends of the get_info chain
    def get_info(self):
        option = self.classics[self.chosen.get()]
        return option['height'], option['width'], option['mines']


# contains custom difficulty manually selected by the player
class CustomDifficulty(tk.Frame):
    def __init__(self, parent, controller, chosen, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.chosen = chosen  # keeps track of which difficulty is currently selected

        radio_button = tk.Radiobutton(self, text='Custom:', variable=self.chosen, value='Custom')
        radio_button.grid(row=0, column=0)

        self.entries = CustomEntries(self, self.controller)  # entries for height/width/mines
        self.entries.grid(row=0, column=1)

        self.density = DensityFrame(self, self.controller)  # buttons for auto-filling the number of mines
        self.density.grid(row=1, column=0)

    def get_info(self):
        return self.entries.get_info()


# contains entries of height/width/mines that the player can input
class CustomEntries(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.height = CustomEntry(self, self.controller, 'height')
        self.height.grid(row=0, column=0)

        self.width = CustomEntry(self, self.controller, 'width')
        self.width.grid(row=0, column=1)

        self.mines = CustomEntry(self, self.controller, 'mines')
        self.mines.grid(row=0, column=2)

    # checks and passes the the input from player (and corrects it if necessary)
    def get_info(self):
        height = self.height.get_info()
        width = self.width.get_info()
        mines = self.mines.get_info()

        if height == 1 and width == 1:  # board is too small
            self.height.delete(0, 'end')
            self.height.insert(0, '2')
            height = 2

        if mines > ((height * width) - 1):  # too many mines for the board's dimensions
            self.mines.delete(0, 'end')
            self.mines.insert(0, str((height * width) - 1))
            mines = (height * width) - 1

        return height, width, mines


# a single entry for one input (either height, width or mines)
class CustomEntry(tk.Entry):
    def __init__(self, parent, controller, data, *args, **kwargs):
        tk.Entry.__init__(self, parent, *args, **kwargs)  # has properties of tk.Entry

        self.controller = controller  # keeps the reference of the MainFrame

        self.data = data  # determines if the entry should contain height, width or mines

        if self.data == 'mines':  # todo: set all to 6 ?
            self.configure(width=6)  # different sizes for different entries
        else:
            self.configure(width=5)

        self.insert(0, self.data)

        self.bind('<Button-1>', self.activate)

    # activates the entry after first click
    def activate(self, _):
        self.bind('<Button-1>', '')
        self.delete(0, 'end')  # deletes pre-entered text

        # now only allows correct inputs
        vcmd = (self.register(self.validate))
        self.configure(validate='key', validatecommand=(vcmd, '%P'))

    # checks if the input is correct (with every change of input (=> after every digit))
    def validate(self, value):
        if str.isdigit(value):  # only allows numbers
            if self.data == 'mines':
                if len(value) < 7 and int(value) != 0:  # limits the length of input
                    return True
            else:
                if len(value) < 4 and int(value) != 0:  # limits the length of input
                    return True
        elif value == '':  # empty string is allowed
            return True

        return False

    # one of the ends of the get_info chain
    def get_info(self):
        if self.get() == self.data or self.get() == '':  # entry is empty or not set
            self.activate(None)
            self.insert(0, '1')  # inserts a default '1'
            # todo: choose a better default value
            return 1
        else:
            return int(self.get())


# option to auto-fill the mines entry from the size (height * width)
class DensityFrame(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        self.label = tk.Label(self, text='Density (optional):')
        self.label.grid(row=0, column=0)

        self.densities = DensityButtons(self, self.controller)  # from for the buttons
        self.densities.grid(row=0, column=1)


# contains the buttons to change the mines entry
class DensityButtons(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)  # has properties of tk.Frame

        self.controller = controller  # keeps the reference of the MainFrame

        # options for different densities, taken from the original game
        self.options = {
            'Easy': {
                'place': 0,
                'quotient': 0.12345  # = 10 / (9 * 9)
            },
            'Medium': {
                'place': 1,
                'quotient': 0.15625  # = 40 / (16 * 16)
            },
            'Hard': {
                'place': 2,
                'quotient': 0.20625  # = 99 / (16 * 30)
            }
        }

        for key in self.options:  # places the buttons
            button = DensityButton(self, self.controller, self.options[key]['quotient'], text=key)
            button.grid(row=self.options[key]['place'], column=0)


# a single button that changes the mines entry
class DensityButton(tk.Button):
    def __init__(self, parent, controller, quotient, *args, **kwargs):
        tk.Button.__init__(self, parent, *args, **kwargs)  # has properties of tk.Button

        self.controller = controller  # keeps the reference of the MainFrame

        self.quotient = quotient

        # todo: enable only after entering the height and width?
        self.configure(command=self.change)

    # inserts the calculated number of mines into the mines entry
    def change(self):
        # using controller: entries = self.controller.screens[MenuFrame].settings.difficulty.options.custom.entries
        entries = self.master.master.master.entries
        if entries.height.get() != 'height' and entries.width.get() != 'width':  # only works when height/width is set
            entries.mines.activate(None)  # (re)activates the mines entry so it could change
            new_mines = str(round(int(entries.height.get()) * int(entries.width.get()) * self.quotient))
            if new_mines == '0':  # can happen from the rounding
                new_mines = '1'
            entries.mines.insert(0, new_mines)


# starts the game after creating the board
class PlayButton(tk.Button):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Button.__init__(self, parent, *args, **kwargs)  # has properties of tk.Button

        self.controller = controller  # keeps the reference of the MainFrame

        self.configure(text='Play', font=('Helvetica', 30))

        self.configure(command=self.play)

    # prepares the board with the current settings and switches screens
    def play(self):
        info = self.master.settings.get_info()  # extracts all the settings from the menu screen
        self.master.master.master.screens[GameFrame].prepare(info)  # resets the old board and creates the new one
        self.master.master.master.show_screen(GameFrame, self.master)  # switches the screens


class BotCheckBox(tk.Checkbutton):#spojit s minescheckboxem
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Checkbutton.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.configure(text='Show Bot')

        self.checked = tk.IntVar()
        self.configure(variable=self.checked)

    def get_info(self):
        if self.checked.get() == 1:
            return True
        else:#asi by slo vynechat
            return False


class MinesCheckBox(tk.Checkbutton):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Checkbutton.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.configure(text='Show mines')

        self.checked = tk.IntVar()
        self.configure(variable=self.checked)

    def get_info(self):
        if self.checked.get() == 1:
            return True
        else:#asi by slo vynechat
            return False


class GameFrame(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.information = InformationPanel(self, self.controller)
        self.information.grid(row=0, column=0)

        self.board = Board(self, self.controller)
        self.board.grid(row=1, column=0)

        self.additional = AdditionalInformation(self, self.controller)

    def prepare(self, info):
        self.information.flags_counter.flags.set(str(info[0][2]))
        self.information.clock.time.set("00:00:00")
        self.board.place_tiles(info[0][0], info[0][1])
        self.board.game = Grid(info[0][0], info[0][1], info[0][2])
        self.board.active = True
        solve(self.board.game)
        if info[1]:
            self.additional.grid(row=2, column=0)
            self.board.show_bot = True
        else:
            self.additional.grid_remove()
            self.board.show_bot = False
        if info[2]:
            self.board.show_mines = True
        else:
            self.board.show_mines = False


class InformationPanel(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.flags_counter = FlagsCounter(self, self.controller)
        self.flags_counter.grid(row=0, column=0)

        self.menu = MenuButton(self, self.controller)
        self.menu.grid(row=0, column=1)

        self.clock = Clock(self, self.controller)
        self.clock.grid(row=0, column=2)


class FlagsCounter(tk.Label):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Label.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.flags = tk.StringVar()
        self.flags.set('0')  #neni potreba ?

        self.configure(textvariable=self.flags)


class MenuButton(tk.Button):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Button.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.configure(text='Menu')

        self.configure(command=self.to_menu)

    def to_menu(self):
        self.controller.show_screen(MenuFrame, self.master.master)
        self.master.master.board.delete_tiles()
        self.master.clock.active = False


class Clock(tk.Label):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Label.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.time = tk.StringVar()
        self.time.set('00:00:00')  #neni potreba ?

        self.active = False

        self.configure(textvariable=self.time)

    def update_clock(self, start_time):
        time_now = time.time()
        seconds_elapsed = int(time_now - start_time)
        time_h = seconds_elapsed // 3600
        time_m = (seconds_elapsed - (time_h * 3600)) // 60
        time_s = seconds_elapsed - (time_h * 3600) - (time_m * 60)
        if time_h < 10:
            hours = '0' + str(time_h)
        else:
            hours = str(time_h)
        if time_m < 10:
            minutes = '0' + str(time_m)
        else:
            minutes = str(time_m)
        if time_s < 10:
            seconds = '0' + str(time_s)
        else:
            seconds = str(time_s)
        if self.active:
            self.time.set(hours + ':' + minutes + ':' + seconds)
            self.controller.master.after(1000, lambda: self.update_clock(start_time))


class Board(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        #self.active = True

        self.tiles = None  # akceptovatelne?

        self.game = None  # -//-

        self.show_bot = False
        self.show_mines = False

        self.coloring = {
            0: {
                'color': 'white',
                'text': ''
            },
            1: {
                'color': 'blue',
                'text': '1'
            },
            2: {
                'color': 'green',
                'text': '2'
            },
            3: {
                'color': 'red',
                'text': '3'
            },
            4: {
                'color': 'purple',
                'text': '4'
            },
            5: {
                'color': 'maroon',
                'text': '5'
            },
            6: {
                'color': 'turquoise',
                'text': '6'
            },
            7: {
                'color': 'black',
                'text': '7'
            },
            8: {
                'color': 'grey',
                'text': '8'
            }
        }

    def place_tiles(self, height, width):
        self.tiles = [[] * width for _ in range(height)]
        for i in range(height):
            for j in range(width):
                tile = TileBlock(self, self.controller)
                self.tiles[i].append(tile)
                tile.grid(row=str(i), column=str(j), padx='1', pady='1')

    def delete_tiles(self):
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[0])):
                self.tiles[i][j].destroy()
        self.tiles = None

    def bot_color(self):
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[0])):
                if self.game.grid[i][j].hidden:
                    if self.game.grid[i][j].info == 'danger':
                        if self.game.grid[i][j].found_in == 'obvious':
                            self.tiles[i][j].configure(background='#990000')
                        elif self.game.grid[i][j].found_in == 'brute':
                            self.tiles[i][j].configure(background='#e60000')
                        elif self.game.grid[i][j].found_in == 'count':
                            self.tiles[i][j].configure(background='#ffb3b3')
                    elif self.game.grid[i][j].info == 'safe':
                        if self.game.grid[i][j].found_in == 'obvious':
                            self.tiles[i][j].configure(background='#006600')
                        elif self.game.grid[i][j].found_in == 'brute':
                            self.tiles[i][j].configure(background='#00ff00')
                        elif self.game.grid[i][j].found_in == 'count':
                            self.tiles[i][j].configure(background='#b3ffb3')

    def mines_color(self):
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[0])):
                if self.game.grid[i][j].hidden:
                    if self.game.grid[i][j].number == -1:
                        self.tiles[i][j].configure(text='(+)')
                    else:
                        self.tiles[i][j].configure(text='')

    def win_check(self):
        if self.game.uncovered == (self.game.height * self.game.width) - self.game.mines_number:
            # print('!!!!!!! WIN !!!!!!!!!')
            return True
        return False

    def show_board(self, char):
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[0])):
                self.tiles[i][j].bind("<Button-1>", '')
                self.tiles[i][j].bind("<Button-2>", '')  # Mac
                self.tiles[i][j].bind("<Button-3>", '')  # Linux, Windows
                if self.game.grid[i][j].hidden:
                    if self.game.grid[i][j].number == -1:
                        self.tiles[i][j].configure(text=char)
                    else:
                        if self.tiles[i][j]['text'] == '!':
                            self.tiles[i][j].configure(text='O')
                        else:  # takes care of '?' and '#' (from show_mines)
                            self.tiles[i][j].configure(text='')

    def show_win(self):
        self.show_board('!')
        timer_info = self.master.information.clock.time.get()
        root_here = self.controller.master
        tk.messagebox.showinfo('Fair Minesweeper', 'You won !!! Your time was: ' + timer_info, parent=root_here)

    def lose(self, row, column):
        self.show_board('(+)')
        self.tiles[row][column].configure(background='black')
        self.master.information.clock.active = False
        root_here = self.controller.master
        root_here.after(100, lambda: tk.messagebox.showinfo('Fair Minesweeper', 'You lost !!!', parent=root_here))


class TileBlock(tk.Label):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Label.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.bind('<Button-1>', self.left_click)
        self.bind('<Button-2>', self.right_click_first)  # for Mac
        self.bind('<Button-3>', self.right_click_first)  # Linux, Windows

        self.configure(width=2, background='gray', foreground='white')

    def left_click(self, event):
        if not self.master.master.information.clock.active:
            self.master.master.information.clock.active = True
            self.master.master.information.clock.update_clock(time.time())

        trigger_row = self.grid_info()['row']
        trigger_col = self.grid_info()['column']
        grid = self.master.game
        grid = player_move(grid, grid.grid[trigger_row][trigger_col])
        grid = solve(grid)  # not necessary

        if self.master.win_check():
            self.master.show_board('!')
            #self.master.active = False
            self.master.master.information.clock.active = False
            root.after(100, self.master.show_win)

        if self.master.active:
            if self.master.show_bot:
                self.master.bot_color()
            if self.master.show_mines:
                self.master.mines_color()

    def right_click_first(self, event):
        self.bind('<Button-1>', '')
        self.bind('<Button-2>', self.right_click_second)  # Mac
        self.bind('<Button-3>', self.right_click_second)  # Linux, Windows
        self.configure(text='!')
        flags = self.master.master.information.flags_counter.flags
        flags.set(str(int(flags.get()) - 1))

    def right_click_second(self, event):
        self.bind('<Button-2>', self.right_click_third)  # Mac
        self.bind('<Button-3>', self.right_click_third)  # Linux, Windows
        self.configure(text='?')
        flags = self.master.master.information.flags_counter.flags
        flags.set(str(int(flags.get()) + 1))

    def right_click_third(self, event):
        self.bind('<Button-1>', self.left_click)
        self.bind('<Button-2>', self.right_click_first)  # Mac
        self.bind('<Button-3>', self.right_click_first)  # Linux, Windows
        self.configure(text='')

    def color(self, number):
        if self['text'] == '!':
            flags = self.master.master.information.flags_counter.flags
            flags.set(str(int(flags.get()) + 1))

        self.configure(text=self.master.coloring[number]['text'])
        self.configure(foreground=self.master.coloring[number]['color'])
        self.configure(background='light grey')
        self.bind('<Button-1>', '')
        self.bind('<Button-2>', '')  # Mac
        self.bind('<Button-3>', '')  # Linux, Windows


class AdditionalInformation(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.controller = controller

        self.lines = {
            1: {
                'text': 'safe found in solve_obvious()',
                'color': '#006600'
            },
            2: {
                'text': 'safe found in solve_brute_force()',
                'color': '#00ff00'
            },
            3: {
                'text': 'safe found in solve_mine_count()',
                'color': '#b3ffb3'
            },
            4: {
                'text': 'mine found in solve_obvious()',
                'color': '#990000'
            },
            5: {
                'text': 'mine found in solve_brute_force()',
                'color': '#e60000'
            },
            6: {
                'text': 'mine found in solve_mine_count()',
                'color': '#ffb3b3'
            }
        }

        for key in self.lines:
            tile = tk.Label(self)
            tile.configure(width=2, background=self.lines[key]['color'])
            tile.grid(row=str(key), column=0)

            label = tk.Label(self)
            label.configure(text=self.lines[key]['text'])
            label.grid(row=str(key), column=1)


def uncover_tile(grid, current_tile):
    if current_tile.hidden:
        if current_tile.number == 0:
            my.screens[GameFrame].board.tiles[current_tile.row][current_tile.col].color(0)
            current_tile.hidden = False
            current_tile.info = 'safe'
            grid.uncovered += 1
            for tile in current_tile.adjacent:
                if tile.hidden:
                    grid = uncover_tile(grid, tile)
        elif current_tile.number == -1:
            my.screens[GameFrame].board.lose(current_tile.row, current_tile.col)
            # print('!!!!!!! LOSE !!!!!!!!')
        else:
            my.screens[GameFrame].board.tiles[current_tile.row][current_tile.col].color(current_tile.number)
            current_tile.hidden = False
            current_tile.pending = True
            current_tile.info = 'safe'
            grid.pending.append(current_tile)
            grid.uncovered += 1
    return grid


root = tk.Tk()
root.resizable(height=False, width=False)
root.title('Fair Minesweeper')
my = MainFrame(root)
my.grid(row=0, column=0)
root.mainloop()
