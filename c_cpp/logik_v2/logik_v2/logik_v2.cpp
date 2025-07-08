// logik_v2.cpp : This file contains the 'main' function. Program execution begins and ends there.


/* General TODO:
    - create Destructors for the classes
    - create conversion table delta->bucket and bucket->delta
*/

/* General Questions:
    - is it better to use printf/scanf versus cin/cout?
    - NULL vs nullptr?
    - really better New than Malloc?
*/


#include <iostream>
// Q: include "<string>" ?
#include <ctime> // Q: optimal choice?


// N ... number of positions, M ... number of colors
int N, M;
// T = M^N ... number of all possible codes, R ... number of all possible comparison answers (deltas)
int T, R;

struct delta {
    int c;
    int d;
};

struct node {
    int value;
    node* next = NULL;
};

int scan_int_parameter(int min_val, int max_val, int default_value, std::string parameter_name);
void scan_code(int* storage);
void print_array(int* storage, int length = N, int row_size = 0);
void guess_code(int* player_code);
delta compare(int* x, int* y, int* cntr_x, int* cntr_y);
int get_T();
int get_R();
int code_to_index(int* storage);
void index_to_code(int index, int* storage);
int delta_to_bucket(delta res);
delta bucket_to_delta(int bucket); // Q: do we even need this function? A1: good for debugging
void print_delta(delta res);



class linked_list {
public:
    node* head = NULL;
    node* tail = NULL;
    int length = 0;
    void add(int value);
    void traverse(bool include_endl=true);
};

void linked_list::add(int value) {
    node* new_node = new node;
    new_node->value = value;
    if (this->length) {
        (this->tail)->next = new_node;
    }
    else {
        this->head = new_node;
    }
    this->tail = new_node;
    this->length += 1;
}

void linked_list::traverse(bool include_endl) {
    node* current_node = this->head;
    while (current_node != nullptr) { // use NULL or nullptr?
        std::cout << current_node->value << " ";
        current_node = current_node->next;
    }
    if (include_endl) {
        std::cout << std::endl;
    }
}



class buckets {
public:
    linked_list** bucket_array;
    int** delta_matrix; // symmetric matrix containing deltas for each combination of code and guess
    int* cntr_x;
    int* cntr_y;
    int* code_x;
    int* code_y;
    enum print_option
    {
        prt_id,
        prt_delta
    };
    buckets() {
        // allocate memory for buckets
        this->bucket_array = new linked_list*[T];
        for (int i = 0; i < T; i++) {
            this->bucket_array[i] = new linked_list[R];
        }

        // allocate memory for the delta matrix
        this->delta_matrix = new int* [T];
        for (int i = 0; i < T; i++) {
            (this->delta_matrix)[i] = new int[T];
        }

        // allocate memory for compare arrays
        this->cntr_x = new int[R];
        this->cntr_y = new int[R];

        // allocate memory for code arrays
        this->code_x = new int[T];
        this->code_y = new int[T];
    }
    void calculate_data();
    void clear_counters();
    void print_delta_matrix(print_option choice);
    void print_buckets(bool print_empty=false);
};

void buckets::clear_counters() {
    for (int i = 0; i < R; i++) {
        (this->cntr_x)[i] = 0;
        (this->cntr_y)[i] = 0;
    }
}

void buckets::calculate_data() {
    delta res;
    int res_index;
    for (int i = 0; i < T; i++) {
        for (int j = i + 1; j < T; j++) { // calculate only the upper diagonal without the diagonal
            index_to_code(i, this->code_x);
            index_to_code(j, this->code_y);
            this->clear_counters();
            res = compare(this->code_x, this->code_y, this->cntr_x, this->cntr_y);
            res_index = delta_to_bucket(res);

            // write into delta matrix
            (this->delta_matrix)[i][j] = res_index;
            (this->delta_matrix)[j][i] = res_index;

            // write into buckets
            ((this->bucket_array)[i][res_index]).add(j);
            ((this->bucket_array)[j][res_index]).add(i);
        }
    }

    res = { N, 0 };
    res_index = delta_to_bucket(res);
    for (int i = 0; i < T; i++) { // calculate diagonal
        (this->delta_matrix)[i][i] = res_index; // "= R-1" is faster but dependent on our implementation of bijection [0,...,R-1] <--> [c,d]
    }
}

void buckets::print_delta_matrix(print_option choice) {
    std::cout << "Delta matrix:" << std::endl;
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            if (choice == prt_id) {
                std::cout << (this->delta_matrix)[i][j] << " ";
            }
            else if (choice == prt_delta) { // change to else?
                print_delta(bucket_to_delta((this->delta_matrix)[i][j]));
            }
        }
        std::cout << std::endl;
    }
}

void buckets::print_buckets(bool print_empty) {
    std::cout << "Buckets:" << std::endl;
    for (int i = 0; i < T; i++) {
        std::cout << "B" << i << ":";
        for (int j = 0; j < R; j++) {
            if (print_empty || ((this->bucket_array)[i][j]).length != 0) {
                print_delta(bucket_to_delta(j));
                std::cout << ": ";
                ((this->bucket_array)[i][j]).traverse(false);
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}



class test_class {
public:
    int* arr;
    test_class(int a) {
        arr = (int*)malloc(2 * sizeof(int));
        arr[0] = a;
        arr[1] = a;
    }
};




int main()
{
    //test_class my_class1(2);
    //printf("%d\n", my_class1.arr[0]);
    //test_class my_class2(3);
    //printf("%d\n", my_class1.arr[0]);

    /*
    linked_list my_ll;
    my_ll.add(69);
    my_ll.add(420);
    my_ll.traverse();
    */
    
    /*
    node node1;
    node node2;
    node1.value = 1;
    node1.next = &node2;
    node2.value = 2;
    node* ptr = &node1;
    while (ptr != NULL) {
        printf("%d ", ptr->value);
        ptr = ptr->next;
    }
    */


    std::cout << "The game LOGIK:" << std::endl;

    // get basic arguments
    N = scan_int_parameter(1, 5, 3, "N");
    M = scan_int_parameter(1, 5, 2, "M");

    // enter player code
    int* player_code = (int*)malloc(N * sizeof(int));
    scan_code(player_code);


    /*
    // testing the functions:
    T = get_T();
    index_to_code(7, player_code);
    print_array(player_code);
    delta test_delta = { 1,2 };
    std::cout << "bucket: " << delta_to_bucket(test_delta) << std::endl;
    std::cout << "c: " << bucket_to_delta(3).c << " d: " << bucket_to_delta(3).d << std::endl;
    */


    // start main algorithm
    guess_code(player_code);

    std::cout << "Program finished correctly." << std::endl;
    return 0;
}

// scans an integer paramater from user - with dialog and restricions based on min/max/default values
int scan_int_parameter(int min_val, int max_val, int default_value, std::string parameter_name) {
    std::cout << "Input the parameter " << parameter_name << " (or input \"0\" for the default value " << default_value << ")." << std::endl;
    int input_val;
    //scanf_s("%d", &input_val);
    std::cin >> input_val;
    if (input_val == 0) {
        return default_value;
    }
    while (input_val < min_val || input_val > max_val) {
        std::cout << "The value must be between " << min_val << " and " << max_val << " ! Insert again:" << std::endl;
        std::cin >> input_val;
    }
    return input_val;
}

// scans a game code of length N restricted on M colors (0 -> M-1)
void scan_code(int* storage) {
    std::cout << "Random (0) or Manual (1)?" << std::endl;
    int choice;
    std::cin >> choice;
    if (choice == 0) {
        std::srand(time(0));
        for (int i = 0; i < N; i++) { // Q: just generate one big number (max equal to T-1) and then convert to code?
            storage[i] = std::rand() % M;
        }
        std::cout << "Code generated:" << std::endl;
        print_array(storage);
    }
    else if (choice == 1) {
        std::cout << "Enter code:" << std::endl;
        int digit;
        bool out_of_bounds = true;

        while (out_of_bounds) {
            out_of_bounds = false;
            for (int i = 0; i < N; i++) {
                std::cin >> digit;
                storage[i] = digit;
                if (digit < 0 || digit >= M) {
                    out_of_bounds = true;
                }
            }
            if (out_of_bounds) {
                std::cout << "Code is incorrect! The colors must be between 0 and " << M - 1 << ". Enter code again:" << std::endl;
            }
        }
    }
}

// prints array, has some additional arguments for formatting
void print_array(int* storage, int length, int row_size) {
    std::cout << "prt: ";
    for (int i = 0; i < length; i++) {
        std::cout << storage[i] << " ";
        if (row_size && (i - 1) % row_size == 0 && i > 0 && i != length - 1) { // TODO: make it more compact
            std::cout << std::endl << "     ";
        }
    }
    std::cout << std::endl;
}

// outputs a tuple of (number of correct positions, number of positions that can be moved to correct position)
delta compare(int* x, int* y, int* cntr_x, int* cntr_y) {
    delta res = {0, 0};
    for (int i = 0; i < N; i++) {
        if (x[i] == y[i]) {
            res.c++;
        }
        else {
            cntr_x[x[i]]++;
            cntr_y[y[i]]++;
        }
    }
    for (int i = 0; i < M; i++) {
        res.d += std::min(cntr_x[i], cntr_y[i]);
    }
    return res;
}

// calculates T iteratively
int get_T() {
    int res = 1;
    for (int i = 0; i < N; i++) {
        res *= M;
    }
    return res;
}

// calculates R by an expression
int get_R() {
    return (N + 1) * (N + 2) / 2;
}

// transform the vector representing the code to a unique integer - calculate it in base M
// least significant M-digit is at the start of the vector ! (e.g. M=2 [0, 1, 1] ~ 6)
int code_to_index(int* storage) {
    int M_power = 1;
    int res = 0;
    for (int i = 0; i < N; i++) {
        res += M_power * storage[i];
        M_power *= M;
    }
    return res;
}

// transform a unique integer to the code that it represents - calculate it in base M
void index_to_code(int index, int* storage) {
    int remainder = index;
    int M_power = T / M;
    for (int i = N - 1; i >= 0; i--) {
        storage[i] = remainder / M_power;
        if (i != 0) {
            remainder = remainder % M_power;
            M_power /= M;
        }
    }
}

// transform delta=[c,d] to a unique integer that will represent it
// use ordering [0,0], [0,1], [1,0], [0,2], [1,1], [2,0], ...
// this yields the formula with res=(sum_{i=1}^{c+d} i) + c
int delta_to_bucket(delta res) {
    return ((res.c + res.d) * ((res.c + res.d) + 1) / 2) + res.c;
}

// transform unique integer into delta=[c,d], use inverse formula to delta_to_bucket()
delta bucket_to_delta(int bucket) {
    //int c_plus_d = floor((sqrt(8 * bucket + 1) - 1));
    int c_plus_d = floor((sqrt(2 * bucket + 0.25) - 0.5));
    //std::cout << "c+d=" << c_plus_d << std::endl;
    int pos = (c_plus_d * (c_plus_d + 1)) / 2;
    //std::cout << "pos=" << pos << std::endl;
    delta res;
    res.c = bucket - pos;
    res.d = c_plus_d - res.c;
    return res;
}

// prints delta in format "[c, d] "
void print_delta(delta res) {
    std::cout << "[" << res.c << ", " << res.d << "] ";
}

// main algorithm
void guess_code(int* player_code) {
    T = get_T();
    R = get_R();

    std::cout << "T: " << T << " R: " << R << std::endl;

    buckets* first_guess = new buckets();
    first_guess->calculate_data();
    first_guess->print_delta_matrix(buckets::prt_delta);
    first_guess->print_buckets();

    /*
    ((first_guess->bucket_array)[1][1]).add(4);
    ((first_guess->bucket_array)[1][1]).add(6);
    ((first_guess->bucket_array)[1][1]).traverse();
    */


}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
