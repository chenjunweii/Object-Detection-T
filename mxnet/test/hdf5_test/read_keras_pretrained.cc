#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <map>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>


using namespace H5;
using namespace std;

herr_t op_func(hid_t loc_id, const char *name, const H5O_info_t *linfo, void *opdata);

int main(){

    string filename = "../../Model/vgg16_weights.h5";

    H5File file(filename.c_str(), H5F_ACC_RDONLY);

    map <string, vector <string>> * dict = new map <string, vector <string>>;

    herr_t status = H5Ovisit(file.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, op_func, (void *)dict);

    cout << "Done ... " << endl;

    status = H5Fclose (file.getId());

    delete dict;

    return 0;

}


herr_t op_func (hid_t loc_id, const char * name, const H5O_info_t * info, void * pdict){

    printf ("/");               /* Print root group in object path */

    map <string, vector <string>> * dict = (map <string, vector <string>> *) pdict;

    hsize_t *num_obj;

    hid_t Gid;

    herr_t status;

    vector <string> *words = new vector <string>;

    string *sp = new string("/");

    int c = 0;



    /*
     * Check if the current object is the root group, and if not print
     * the full path name and type.
     */

    if (name[0] == '.')         /* Root group, do not print '.' */
        printf ("  (Group)\n");
    else if (info->type == H5O_TYPE_GROUP){

        printf ("%s  (Group)\n", name);

        Gid = H5Gopen1(loc_id, name);

        H5Gget_num_objs(Gid, num_obj);

        (*dict)[name] = vector <string> (*num_obj);

        for (int j = 0; j != (*num_obj); j++)

            (*dict)[name][j] = "None";

        delete num_obj;

    }

    else if (info->type == H5O_TYPE_DATASET){

        printf ("%s  (Dataset)\n", name);

        boost::split((*words), name, boost::is_any_of(*sp));

        while (true){

            if ((*dict)[(*words)[0]][c] == "None"){

                (*dict)[(*words)[0]][c] = (*words)[1];

                break;

            }

            else

                c += 1;
        }

        delete words; delete sp;

    }

    else if (info->type == H5O_TYPE_NAMED_DATATYPE){

        printf ("%s  (Datatype)\n", name);

    }

    else

        printf ("%s  (Unknown)\n", name);


    return 0;
}
