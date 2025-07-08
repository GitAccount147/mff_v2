#include <iostream>
int read_int()
{
    //todo sezer neciselne znaky
    int number = 0;
    char c = getchar();
    while (c >= '0' && c <= '9')
    {
        number = 10 * number + c - '0';
        c = getchar();
    }
    return number;
}
typedef struct ll_cells
{
    int* coordinates;
    struct ll_cells* next;
} ll_cell;
struct vector
{
    int* elements;
};
class linked_list
{
    linked_list(int n)
    {
        this->head = new ll_cell;
        head->next = NULL;
        this->last = this->head;
        this->n = n;
    }
    void add_cell_at_end(int n,  int* coordinates)
    {
        ll_cell* new_cell = new ll_cell;
        int* cell_coordinates = new int[n];
        for (int i = 0; i < n; i++)
        {
            cell_coordinates[i] = coordinates[i];
        }
        new_cell->coordinates = cell_coordinates;
        new_cell->next = NULL;
        this->last->next = new_cell;
    }
    int* pop_cell()
    {
        
        if (this->head->next != NULL)
        {
            ll_cell* temp = this->head->next;
            int* cell_coordinates = new int[this->n];
            for (int i = 0; i < this->n; i++)
            {
                cell_coordinates[i] = temp->coordinates[i];
            }
            this->head->next = temp->next;
            delete(temp);
            return cell_coordinates;
        }
        else
        {
            return NULL;
        }
    }
private:
    ll_cell* head;
    ll_cell* last;
    int n;
};
class md_array
{
public:
    int n;
    int* dimensions;
    md_array(int n, int* dimensions)
    {
        this->n = n;
        this->dimensions = dimensions;
        this->base = this->get_base(n, dimensions);
        this->data = new int[(this->base[n - 1]) * dimensions[n - 1]];
        printf("data size:%d\n", (this->base[n - 1]) * dimensions[n - 1]);
    }
    int* get_cell(int* coordinates)
    {
        int position = 0;
        for (int i = 0; i < this->n; i++)
        {
            printf("{pos: %d cor: %d bas: %d}\n", position, coordinates[i], this->base[i]);
            position = position + coordinates[i] * this->base[i];
        }
        return &(data[position]);
    }
    void pre_fill(int default_value)
    {
        for (int i = 0; i < (this->base[n - 1]) * dimensions[n - 1]; i++)
        {
            this->data[i] = default_value;
        }
    }
private:
    int* data;
    int* base;
    int* get_base(int n, int * dimensions)
    {
        int number = 1;
        int* base = new int[n];
        for (int i = 0; i < n; i++)
        {
            base[i] = number;
            printf("(%d)", base[i]);
            number = number * dimensions[i];
        }
        return base;
    }
};
class Object
{
public:
    int n;
    int* position;
    int* dims;
    vector* valid_vectors;
    Object(int n, int* position, int* dims)
    {
        this->n = n;
        this->position = new int[n];
        this->dims = new int[n];
        for (int i = 0; i < n; i++)
        {
            this->position[i] = position[i];
            this->dims[i] = dims[i];
        }
    }
    int check_boundaries(int* vector)
    {
        for (int i = 0; i < n; i++)
        {
            if (!(this->position[i] + vector[i] > 0 && this->position[i] + vector[i] < this->dims[i]))
            {
                return 0;
            }
        }
        return 1;
    }
    virtual void get_valid_vectors(vector* all_vectors, int all_vectors_size) = 0;
    virtual int specific_check() = 0;
};
class King : public Object
{
    void get_valid_vectors(vector* all_vectors, int all_vectors_size)
    {
        vector* temp = new vector[all_vectors_size];//jsem linej to delat pres spojak
        int valid_vectors_count = 0;
        for (int i = 0; i < all_vectors_size; i++)
        {
            int ones_count = 0;
            int zeros_count = 0;
            for (int j = 0; j < n; j++)
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
            if (n == ones_count + zeros_count && zeros_count != n && ones_count <= 2)
            {
                for (int j = 0; j < n; j++)
                {
                    temp[valid_vectors_count].elements[j] = all_vectors[i].elements[j];
                }
                valid_vectors_count++;
            }
        }
        this->valid_vectors = new vector[valid_vectors_count];
        for (int i = 0; i < valid_vectors_count; i++)
        {
            this->valid_vectors[i] = temp[i];
        }
        delete(temp);//snad se to nezkurvi
    }
    int specific_check()
    {
        return 1;
    }
};
typedef struct ll_o_cells
{
    Object* obj;
    struct ll_o_cells* next;
} ll_o_cell;
class problem_solver 
{
private:
    md_array* state_space;
    int* start;
    int* finish;
    ll_o_cell* objects
    /*
    enum general_conditions {KingGen, WaterGen};
    general_conditions gen_cond = KingGen;
    enum instant_conditions { KingIns, WaterIns };
    instant_conditions ins_cond = KingIns;
    int check_cond(int condition_type)//udelej pres pointery na fce
    {
        switch (condition_type)
        {
        case KingGen:
            return this->gen_condition_king();
        case WaterGen:
            return this->gen_condition_water();
        }
    }
    int gen_condition_king()
    {

    }
    int gen_condition_water()
    {

    }
    */
    vector* all_vectors;
    vector* current_vector;
    int vector_counter = 0;
    int all_vectors_size;
    void vector_generator(int n, int* dims)//n, dims brat z md_array
    {
        this->all_vectors_size = 1;
        for (int i = 0; i < n; i++)
        {
            this->all_vectors_size *= 2 * dims[i] - 1;
        }
        this->all_vectors_size++;
        this->all_vectors = new vector[this->all_vectors_size];
        //printf("all vectors size: %d\n", size);
        for (int i = 0; i < this->all_vectors_size; i++)
        {
            this->all_vectors[i].elements = new int[n];
        }
        this->current_vector = new vector;
        this->current_vector->elements = new int[n];
        this->recursor(0);
    }
    void recursor(int depth)
    {
        if (depth == this->state_space->n)
        {
            for (int i = 0; i < this->state_space->n; i++)
            {
                this->all_vectors[this->vector_counter].elements[i] = this->current_vector->elements[i];
                //printf("(%d) ", this->all_vectors[this->vector_counter].elements[i]);
            }
            //printf("\n");
            this->vector_counter++;
        }
        else
        {
            for (int i = 1 - this->state_space->dimensions[depth]; i < this->state_space->dimensions[depth]; i++)
            {
                this->current_vector->elements[depth] = i;
                this->recursor(depth + 1);
            }
        }
    }
    void select_valid()
    {
        for (int i = 0; i < this->all_vectors_size; i++)
        {

        }
    }
    void load_data()
    {
        this->get_obstacles(-2);
        this->start = this->read_one_cell();
        this->finish = this->read_one_cell();
    }
    void get_obstacles(int tag)//dodelat u vsech minus 1, aby to bylo 0 ... (n-1)
    {
        int obstacle_count = read_int();
        int* obstacle_coordinates = new int[this->state_space->n];
        for (int i = 0; i < obstacle_count; i++)
        {
            for (int j = 0; j < this->state_space->n; j++)
            {
                obstacle_coordinates[j] = read_int();
            }
            *(this->state_space->get_cell(obstacle_coordinates)) = tag;
        }
    }
    int* read_one_cell()
    {
        int* cell = new int[this->state_space->n];
        for (int i = 0; i < this->state_space->n; i++)
        {
            cell[i] = read_int();
        }
        return cell;
    }
public:
    problem_solver(int dimensions, int* sizes)
    {
        this->state_space = new md_array(dimensions, sizes);
        this->state_space->pre_fill(-1);
        this->load_data();
        this->vector_generator(this->state_space->n, this->state_space->dimensions);
    }
    void solve()
    {

    }
};
int main()
{
    int my_dims[3] = { 2, 3, 5 };
    problem_solver* my_solver = new problem_solver(3, my_dims);
    return 0;
}