// C++ program to demonstrate various function of 
// unordered_multimap 
#include <bits/stdc++.h> 
using namespace std; 
  
//making typedef for short declaration 
typedef unordered_multimap<string, int>::iterator umit; 
  
// Utility function to print unordered_multimap 
void printUmm(unordered_multimap<string, int> umm) 
{ 
    // begin() returns iterator to first element of map 
    umit it = umm.begin(); 
  
    for (; it != umm.end(); it++) 
        cout << "<" << it->first << ", " << it->second 
             << ">  "; 
  
    cout << endl; 
} 
  
// Driver program to check all function 
int main() 
{ 
    // empty initialization 
    unordered_multimap<string, int> umm1; 
  
    // Initialization bu intializer list 
    unordered_multimap<string, int> umm2 ({{"10", 1}, 
                                           {"15",40},
                                           {"10", 2}, 
                                           {"11", 10}, 
                                           {"13", 7}, 
                                           {"14", 9}, 
                                           {"15", 6}, 
                                           {"11", 1}}); 
  
    // Initialization by assignment operation 
    umm1 = umm2; 
    printUmm(umm1); 
  
    // empty returns true, if container is empty else it returns 
    // false 
    if (umm2.empty()) 
        cout << "unordered multimap 2 is empty\n"; 
    else
        cout << "unordered multimap 2 is not empty\n"; 
  
    // size returns total number of elements in container 
    cout << "Size of unordered multimap 1 is " << umm1.size() 
         << endl; 
  
    string key = "15"; 
  
    // find and return any pair, associated with key 
    umit it = umm1.find(key); 
    if (it != umm1.end()) 
    { 
        cout << "\nkey " << key << " is there in unordered "
             << " multimap 1\n"; 
        cout << "\none of the value associated with " << key 
             << " is " << it->second << endl; 
    } 
    else
        cout << "\nkey " << key << " is not there in unordered"
             << " multimap 1\n"; 
  
    // count returns count of total number of pair associated 
    // with key 
    int cnt = umm1.count(key); 
    cout << "\ntotal values associated with " << key << " are "
         << cnt << "\n\n"; 
  
    printUmm(umm2); 
  
    // one insertion by makeing pair explicitly 
    umm2.insert(make_pair("13", 11)); 
  
    // insertion by initializer list 
    umm2.insert({{"5", 12}, {"0", 33}}); 
    cout << "\nAfter insertion of <5, 12> and <0, 33>\n"; 
    printUmm(umm2); 
  
    // erase deletes all pairs corresponding to key 
    umm2.erase("13"); 
    cout << "\nAfter deletion of apple\n"; 
    printUmm(umm2); 
  
    // clear deletes all pairs from container 
    umm1.clear(); 
    umm2.clear(); 
  
    if (umm2.empty()) 
        cout << "\nunordered multimap 2 is empty\n"; 
    else
        cout << "\nunordered multimap 2 is not empty\n"; 
} 