#include <iostream>
#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>
#include <rapidxml/rapidxml_print.hpp>

using namespace std;


namespace flt{

	namespace fxml{

		rapidxml::xml_node <> * find_sibling_node(rapidxml::xml_node <> *, string); // find next sibling node, return node if found else nullptr;
		rapidxml::xml_node <> * find_child_node(rapidxml::xml_node <> *, string); // find next child node, return node if found else nullptr;
		rapidxml::xml_node <> * load_xml(string); // return root node
	
	} /* fxml */

} /* flt */
