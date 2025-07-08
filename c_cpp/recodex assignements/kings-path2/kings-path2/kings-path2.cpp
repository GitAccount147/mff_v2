/*
* poznamky:
*  - ano, jsem si vedom ze jsem to asi s obecnosti implementace asi prehnal ale snad to praci naopak usnadni
*  - psat kod v cestine mi prijde jako prasarna proto v anglictine (ta asi nebude perfektni ale i tak)
*  - urcite to neni nejlepsi objektovy navrh ale koneckoncu to ma byt na procviceni
*  - spojak by asi sel napsat pro obecny typ ale jelikoz stejne budou jen 2 tak mi to prislo zbytecne
*  - napady na implementaci zmen v zadani:
*    - bude vice kralu:
*      - staci dopsat do Simulator->load_data() vice instanci
*    - bude jina figurka:
*      - staci napsat (popr. prepsat) obdobnou classu jako King (i se statickymi promenymi)
*      - dodelat Object->specific_check() je-li treba
*      - dopsat do Simulator->load_data()
*    - bude vice ruznych figurek a jejich instanci:
*      - obdobne + doimplementovat ze figurka nemuze skocit na jinou (treba nastavit na tyto pozice take (-2))
*    - sachovnice nebude 8x8:
*      - v mainu zmenit 2 cisla
*    - nebude uz to sachovnice, a treba nebude ani 2d:
*      - v mainu zmenit 2 cisla
*      - udelat specifickou classu pro novy problem ktery resime
*    - nacitani nebude z konzole ale treba ze souboru
*      - zmenit Simulator->load_data()
*    - ve vstupu bude vetsi neporadek
*      - lepe ocistovat v read_int() nebo Simulator->load_data()
*/

#include <iostream>

int read_int() // reads a non-negative integer
{
    int number = 0;
    char c = getchar();
    while (!(c >= '0' && c <= '9')) // gets rid of non-digit chars
    {
        c = getchar();
    }
    while (c >= '0' && c <= '9')
    {
        number = 10 * number + c - '0';
        c = getchar();
    }
    return number;
}

struct array // for handling 2d arrays more easily
{
    int* elements;
};

typedef struct ll_ip_cells // ~linked_list_integer_pointer_cell; single cell
{
    int* elements;
    int depth;
    struct ll_ip_cells* next;
} ll_ip_cell;

class Linked_list_int_p // linked list class for managing the breadth-first-search
{
public:
    Linked_list_int_p()
    {
        this->head = new ll_ip_cell;
        head->next = NULL;
        this->last = this->head;
    }

    void add_cell_at_end(int* coordinates, int depth)
    {
        ll_ip_cell* new_cell = new ll_ip_cell;
        new_cell->elements = coordinates;
        new_cell->depth = depth;
        new_cell->next = NULL;

        if (this->head->next == NULL) // list is empty
        {
            this->head->next = new_cell;
            this->last = new_cell;
        }
        else
        {
            this->last->next = new_cell;
            this->last = new_cell;
        }
    }

    ll_ip_cell* pop_cell()
    {
        if (this->head->next != NULL)
        {
            ll_ip_cell* temp = this->head->next;
            this->head->next = temp->next;
            return temp;
        }
        else
        {
            return NULL;
        }
    }
private:
    ll_ip_cell* head;
    ll_ip_cell* last; // for easier adding to the queue
};

class Md_array // generic multidimensional array
{
public:
    int dimensions; // number of dimensions
    int* ranges; // ranges of the arrays
    Md_array(int dimensions, int* ranges)
    {
        this->dimensions = dimensions;
        this->ranges = ranges;
        this->get_base(dimensions, ranges);
        this->data = new int[this->data_size];
    }

    int* get_cell(int* coordinates) // returns pointer to a position
    {
        int position = 0; // internal linear value
        for (int i = 0; i < this->dimensions; i++)
        {
            position = position + coordinates[i] * this->base[i];
        }
        return &(data[position]);
    }

    void fill(int default_value) // fills all the positins with a default value
    {
        for (int i = 0; i < this->data_size; i++)
        {
            this->data[i] = default_value;
        }
    }

    void refill(int value) // for multiple uses of the array
    {
        for (int i = 0; i < this->data_size; i++)
        {
            if (this->data[i] != -2)
            {
                this->data[i] = value;
            }
        }
    }

    bool check_boundaries(int* position) // checks if the position is inside the boundaries of the array
    {
        for (int i = 0; i < this->dimensions; i++)
        {
            if (!(position[i] > -1 && position[i] < this->ranges[i]))
            {
                return false;
            }
        }
        return true;
    }

private:
    int* data; // lineary stored data
    int data_size; // number of positions
    int* base; // newly created base to access members by position

    void get_base(int dimensions, int* ranges) // creates the base, finds the size of the data
    {
        int* base = new int[dimensions];
        base[0] = 1;
        for (int i = 1; i < dimensions; i++)
        {
            base[i] = base[i - 1] * ranges[i - 1]; // base values increase with the values of ranges
            //printf("(%d)", base[i]);
        }
        this->data_size = base[dimensions - 1] * ranges[dimensions - 1];
        this->base = base;
    }
};

class Object // generic object inside the simulation (abstract class)
{
public:
    // static variables for initialization without any instance Object:
    static int dimensions;
    static int* ranges;
    static array* all_vectors; // stores all possible vectors ( ~ directions) that the Object instance can move with
    static int all_vectors_size;
    static array* current_vector; // for recursive method
    static int vector_counter; // for recursive method

    int* position; // position inside the md_array
    int* destination; // how many moves does it take to get there

    // static methods for initialization without any instance Object:
    static void initialize(int dimensions, int* ranges);
    static void generate_all_vectors(); // generates all the possible moves (~vectors) for the state space (~all pairs of positions in ht earray)
    static void recursor(int depth); // recursive function for generating all vectors
    virtual void get_valid_vectors() = 0; // gets vectors specific for the class-child
    virtual bool specific_check() = 0; // specific check for dynamic conditions (e.g. Queen cannot move over other chess pieces)
    virtual array* pass_valid_vectors() = 0;
    virtual int pass_valid_vectors_count() = 0;
};

// declaration of static variables for the Object class:
int Object::dimensions = 0;
int* Object::ranges = NULL;
array* Object::all_vectors = NULL;
int Object::all_vectors_size = 0;
array* Object::current_vector = NULL;
int Object::vector_counter = 0;

// declaration of static methods for the Object class:
void Object::initialize(int dimensions_input, int* ranges_input) // sets the main variables
{
    dimensions = dimensions_input;
    ranges = new int[dimensions];
    for (int i = 0; i < dimensions; i++)
    {
        ranges[i] = ranges_input[i];
    }
    generate_all_vectors();
}

void Object::generate_all_vectors() // generates all the possible moves (~vectors) for the state space (~all pairs of positions in ht earray)
{
    all_vectors_size = 1;
    for (int i = 0; i < dimensions; i++)
    {
        all_vectors_size *= 2 * ranges[i] - 1; // counts all possible differences between two cells in the md_array
    }
    all_vectors_size++; // includes the zero vector (0 0 ... 0 0)

    all_vectors = new array[all_vectors_size];
    for (int i = 0; i < all_vectors_size; i++)
    {
        all_vectors[i].elements = new int[dimensions];
    }

    current_vector = new array;
    current_vector->elements = new int[dimensions];
    recursor(0); // recursive method that fills up the prepared storage with specific vector values
}

void Object::recursor(int depth) // recursive function for generating all vectors
{
    if (depth == dimensions)
    {
        for (int i = 0; i < dimensions; i++)
        {
            all_vectors[vector_counter].elements[i] = current_vector->elements[i];
        }
        vector_counter++;
    }
    else
    {
        for (int i = 1 - ranges[depth]; i < ranges[depth]; i++) // branches the recursion
        {
            current_vector->elements[depth] = i;
            recursor(depth + 1);
        }
    }
}

typedef struct ll_obj_cells // ~linked_list_object_cell; single cell
{
    Object* instance;
    struct ll_obj_cells* next;
} ll_obj_cell;

class Linked_list_obj // linked list for storing the object instances inside the simulation
{
public:
    ll_obj_cell* head;
    Linked_list_obj()
    {
        this->head = new ll_obj_cell;
        head->next = NULL;
        this->last = this->head;
    }
    void add_cell_at_end(Object* instance)
    {
        ll_obj_cell* new_cell = new ll_obj_cell;
        new_cell->instance = instance;
        new_cell->next = NULL;

        if (this->head->next == NULL) // list is empty
        {
            this->head->next = new_cell;
            this->last = new_cell;
        }
        else
        {
            this->last->next = new_cell;
            this->last = new_cell;
        }
    }
    // pop_cell() not implemented since the object(s) cannot dissapear (yet)
private:
    ll_obj_cell* last;
};

class King : public Object // derived class specifying a (chess) kings behaviour
{
public:
    static int number_of_instances;
    static array* valid_vectors; // stores vectors specific to this class
    static int valid_vectors_count;

    King(int* position, int* destination)
    {
        this->position = new int[dimensions];
        this->destination = new int[dimensions];
        for (int i = 0; i < this->dimensions; i++)
        {
            this->position[i] = position[i];
            this->destination[i] = destination[i];
        }

        if (number_of_instances == 0) // if this is the first instance of the object it generates the valid vectors
        {
            this->get_valid_vectors();
        }
        number_of_instances++;
    }

    bool specific_check() // there are no specific boundaries of movement for this piece
    {
        return true;
    }

    array* pass_valid_vectors()
    {
        return valid_vectors;
    }

    int pass_valid_vectors_count()
    {
        return valid_vectors_count;
    }

private:
    void get_valid_vectors() // creates vectors (~moves) specific to this class
    {
        array* temporary = new array[all_vectors_size];
        for (int i = 0; i < all_vectors_size; i++)
        {
            temporary[i].elements = new int[dimensions];
        }

        // logic of the kings movement:
        int ones_count, zeros_count;
        for (int i = 0; i < all_vectors_size; i++)
        {
            ones_count = 0; // counter for (-1)s and (1)s
            zeros_count = 0;
            for (int j = 0; j < dimensions; j++)
            {
                if (all_vectors[i].elements[j] == 0)
                {
                    zeros_count++;
                }
                else if (all_vectors[i].elements[j] == 1 || all_vectors[i].elements[j] == -1)
                {
                    ones_count++;
                }
            }

            if (dimensions == ones_count + zeros_count && zeros_count != dimensions && ones_count <= 2) // main if
            {
                for (int j = 0; j < dimensions; j++)
                {
                    temporary[valid_vectors_count].elements[j] = all_vectors[i].elements[j];
                }
                valid_vectors_count++;
            }
        }

        valid_vectors = new array[valid_vectors_count];
        for (int i = 0; i < valid_vectors_count; i++)
        {
            valid_vectors[i] = temporary[i];
        }
        delete(temporary);//snad se to nezkurvi
    }
};

// declaration of static variables for the King class
array* King::valid_vectors = NULL;
int King::number_of_instances = 0;
int King::valid_vectors_count = 0;

class Simulator // main class containing the specific array and all the objects inside of it
{
public:
    Simulator(int dimensions, int* ranges)
    {
        // prepares the simulation:
        this->state_space = new Md_array(dimensions, ranges);
        this->objects = new Linked_list_obj();
        this->state_space->fill(-1); // prefill the array with (-1)s
        Object::initialize(dimensions, ranges);
        this->load_data();
    }
    void solve() // main simulation loop
    {
        ll_obj_cell* current_object_cell = this->objects->head->next; // starts with the first object in the linked list
        Object* current_object;

        while (current_object_cell != NULL) // while there are still some objects inside the linked list
        {
            current_object = current_object_cell->instance;
            this->state_space->refill(-1); // set for finding out the lengths of paths separately
            this->bfs_list = new Linked_list_int_p();

            ll_ip_cell* cell = new ll_ip_cell(); // the starting position of the BFS
            cell->depth = 1;
            cell->elements = current_object->position;
            *(this->state_space->get_cell(cell->elements)) = 0;

            while (cell != NULL) // while there are still positions in the queue
            {
                array* current_vectors = current_object->pass_valid_vectors();
                for (int i = 0; i < current_object->pass_valid_vectors_count(); i++) // loops through the valid vectors of the current object
                {
                    int* new_position = new int[this->state_space->dimensions]; // new position "where the object will be moved with the vector"
                    for (int j = 0; j < this->state_space->dimensions; j++)
                    {
                        new_position[j] = cell->elements[j] + current_vectors[i].elements[j];
                    }

                    if (this->state_space->check_boundaries(new_position)) // is inside the array
                    {
                        if (current_object->specific_check()) // checks th dynamic condition
                        {
                            if (*(this->state_space->get_cell(new_position)) == -1) // "we haven't been here" and it isnt an obstacle (-2)
                            {
                                this->bfs_list->add_cell_at_end(new_position, (cell->depth) + 1); // adds new position to the queue
                                *(this->state_space->get_cell(new_position)) = cell->depth;
                            }
                        }
                    }
                }
                cell = this->bfs_list->pop_cell(); // grabs next in queue
            }

            printf("%d", *(this->state_space->get_cell(current_object->destination)));

            current_object_cell = current_object_cell->next; // grabs next in linked list
        }
    }

private:
    Md_array* state_space; // multi-dimensional array instance, stores info about the BFS
    Linked_list_obj* objects; // linked list with all the objects inside the simulation
    Linked_list_int_p* bfs_list; // linked list of positions for the BreadthFirstSearch

    void load_data() // loads all obstacles and objects
    {
        this->get_obstacles(-2); // blocks with obstacles will be represented with a (-2)

        // implementation of adding more objects depends on the input style; adds 1 king on the board:
        int objects_to_add = 1;
        int object_identifier = 1;
        int* position, * destination;
        for (int i = 0; i < objects_to_add; i++)
        {
            position = this->read_one_cell();
            destination = this->read_one_cell();
            switch (object_identifier)
            {
            case 1:
                this->objects->add_cell_at_end(new King(position, destination));
                break;
            }
        }
    }

    void get_obstacles(int tag) // reads coordinates of obstacles and inserts a number there
    {
        int obstacle_count = read_int();
        int* obstacle_coordinates = new int[this->state_space->dimensions]; // single obstacle
        for (int i = 0; i < obstacle_count; i++)
        {
            for (int j = 0; j < this->state_space->dimensions; j++)
            {
                obstacle_coordinates[j] = read_int() - 1; // input coordinates start with 1 !!!
            }
            *(this->state_space->get_cell(obstacle_coordinates)) = tag;
        }
    }

    int* read_one_cell() // reads coordinates of a single cell
    {
        int* cell = new int[this->state_space->dimensions];
        for (int i = 0; i < this->state_space->dimensions; i++)
        {
            cell[i] = read_int() - 1; // input coordinates start with 1 !!!
        }
        return cell;
    }
};
int main()
{
    int chess_board[2] = { 8, 8 };
    Simulator* my_simulator = new Simulator(2, chess_board);
    my_simulator->solve();
    return 0;
}