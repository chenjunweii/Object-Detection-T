#ifndef UTILS_HH
#define UTILS_HH

#include <memory>
#include <assert.h>
#include <math.h>
#include <mutex>
#include <unistd.h>

inline int clip(int x, int lower, int upper) {
  
	return max(lower, min(x, upper));
}

template <class T>

inline void acquire(T & out, deque <T> & q, mutex & m, bool & get, unsigned int wait, string comment){

	get = false;

	//printf("[*] Enter %s\n", comment.c_str());
	
	while (! get){

		while (q.empty()){
			
			//printf("[*] %s Queue is not enough : %d\n", comment.c_str(), q.size());

			usleep(wait);
		}

		get = m.try_lock();
	}

	out = move(q.front());

	q.pop_front();

	m.unlock();
	
	q.shrink_to_fit();

	get = false;
	
	usleep(wait);

}

template <class T>

inline void acquire(vector <T> & out, deque <T> & q, mutex & m, bool & get, int nbatch, unsigned int wait, string comment){

	get = false;

	//printf("[*] Enter %s\n", comment.c_str());
	
	while (! get){

		while (q.empty() || (q.size() < nbatch)){
			
			//printf("[*] %s Queue is not enough : %d\n", comment.c_str(), q.size());

			usleep(wait);
		}

		get = m.try_lock();
	}

	for (int b = 0; b != nbatch; ++b){

		out.emplace_back(move(q.front()));
	
		q.pop_front();

	}
	
	m.unlock();
	
	q.shrink_to_fit();

	get = false;
	
	usleep(wait);

}

template <class T>

inline void emplace(vector <T> & in, deque <T> & q, mutex & m, bool & get, int nbatch, unsigned int wait){
	
	get = false;
	
	while (! get){

		get = m.try_lock();
		
		if (not get)
			
			usleep(wait);
	}

	for (int b = 0; b != nbatch; ++b)

		q.emplace_back((in[b]));
	
	m.unlock();

	in.clear();

	in.shrink_to_fit();
	
	get = false;

	usleep(wait);

	
}

template <class T>

inline void emplace(T & in, deque <T> & q, mutex & m, bool & get, unsigned int wait, bool reuse){
	
	get = false;
	
	while (! get){

		get = m.try_lock();
		
		if (not get)
			
			usleep(wait);
	}

	if (reuse)

		q.emplace_back(in);

	else

		q.emplace_back(move(in));
	
	m.unlock();

	get = false;

	usleep(wait);

	
}

#endif
