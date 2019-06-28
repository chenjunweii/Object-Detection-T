#ifndef FLT_XML_HH
#define FLT_XML_HH

#include <iostream>
#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>
#include <rapidxml/rapidxml_print.hpp>
#include <boost/filesystem.hpp>
#include "src/xml.h"

using namespace std;

inline rapidxml::xml_node <> * flt::fxml::find_child_node(rapidxml::xml_node <> * node, string name){

	rapidxml::xml_node <> * cursor = node->first_node();

	//cout << "current name : " << cursor->name() << endl;

	//cout << "target name : " << name << endl;

	if(cursor->name() == name)

		return cursor;

	while(true){

		cursor = cursor->next_sibling();

		if (cursor == 0)

			break;

		else if (cursor->name() == name)

			return cursor;

	}

	return nullptr;
}

inline rapidxml::xml_node <> * flt::fxml::find_sibling_node(rapidxml::xml_node <> * node, string name){

	//cout << endl << "Find Sibling ..." << endl;

	rapidxml::xml_node <> * cursor = node;

	while(true){

		cursor = cursor->next_sibling();

		if (cursor == 0)

			break;

		else if (cursor->name() == name){

			//cout << "name == name" << endl;

			//cout << "Cursor : " << cursor->name() << endl;

			//cout << cursor->

			//cout << "exit find sibling..." << endl;

			return cursor;

		}

	}

	return nullptr;
}

inline rapidxml::xml_node <> * flt::fxml::load_xml(string filename){

	if (!boost::filesystem::exists(filename))

		return nullptr;

	rapidxml::file <> fdoc(filename.c_str());

	rapidxml::xml_document <> doc;

	doc.parse <0> (fdoc.data());

	return doc.first_node();
}

#endif
