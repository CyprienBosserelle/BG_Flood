#ifndef UTCTIME_H
#define UTCTIME_H

#include "General.h"
#include "ReadInput.h"

struct UTCClock
{
	typedef std::chrono::microseconds duration;
	typedef duration::rep rep;
	typedef duration::period period;
	typedef std::chrono::time_point<UTCClock, duration> time_point;
	static const bool is_steady = true;

	//      
	// every time_point will be generated from here
	//
	static time_point fromDate(int year = 0, int month = 0, int day = 0,
		int hour = 0, int min = 0, int sec = 0,
		int usec = 0);
	//
	// convert time_point to a date/time representation
	//
	static void toDate(const time_point& tp,
		int& year, int& month, int& day,
		int& hour, int& min, int& sec,
		int& usec);

	// NOT Supported, we don't need current time. We only
	// want to represent UTC DateTime
	// static time_point now(); 
};

using UTCTime = std::chrono::time_point<UTCClock, std::chrono::microseconds>;

long long date_string_to_time(std::string date);
double date_string_to_s(std::string datetime, std::string refdate);
double readinputtimetxt(std::string input, std::string & refdate);
bool testime1(int hour);
bool testime2(int hour);

#endif
