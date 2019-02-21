#include <iostream>
#include <string>
#include <H5Cpp.h>

using namespace H5;
using namespace std;

herr_t op_func (hid_t loc_id, const char *name, const H5O_info_t *info,
            void *operator_data);

int main(){

  string filename = "test_create.h5";

  herr_t status;

  hid_t file = H5Fopen (filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);


  printf ("Objects in the file:\n");

  status = H5Ovisit (file, H5_INDEX_NAME, H5_ITER_NATIVE, op_func, NULL);

}

herr_t op_func (hid_t loc_id, const char *name, const H5O_info_t *info,
            void *operator_data)
{
    printf ("/");               /* Print root group in object path */

    /*
     * Check if the current object is the root group, and if not print
     * the full path name and type.
     */
    if (name[0] == '.')         /* Root group, do not print '.' */
        printf ("  (Group)\n");
    else
        switch (info->type) {
            case H5O_TYPE_GROUP:
                printf ("%s  (Group)\n", name);
                break;
            case H5O_TYPE_DATASET:
                printf ("%s  (Dataset)\n", name);
                DataSet dataset = file.openDataSet( DATASET_NAME );
                //H5Dopen(loc_id, name, loc_id);
                break;
            case H5O_TYPE_NAMED_DATATYPE:
                printf ("%s  (Datatype)\n", name);
                break;
            default:
                printf ("%s  (Unknown)\n", name);


        }

    return 0;
}
