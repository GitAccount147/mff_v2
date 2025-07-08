#include <iostream>
class sachovnice {        
public:
    // 2d pole
    // instance fronty
    void nacti_ze_vstupu();
    void projdi_do_sirky();
    void vypis_vysledky();
};
class fronta {
public:
    void pridej();
    void uber();
};
int main()
{
    sachovnice moje_sachovnice = new sachovnice;
    moje_sachovnice.nacti_ze_vstupu();
    moje_sachovnice.projdi_do_sirky();
    moje_sachovnice.vypis_vysledky();
}