
#include "utctime.h"

namespace chrono = std::chrono;
using chrono::duration_cast;
using chrono::time_point_cast; 
/*
namespace {

std::time_t to_time_t(const UTCClock::time_point &tp) noexcept
{
  return std::time_t(
      duration_cast<chrono::seconds>(tp.time_since_epoch()).count());
}

UTCClock::time_point from_time_t(std::time_t tt) noexcept
{
  return time_point_cast<UTCClock::duration>(
     chrono::time_point<UTCClock,chrono::seconds>(chrono::seconds(tt)));
}

} // namespace
*/
// Algorithm: http://howardhinnant.github.io/date_algorithms.html
int days_from_epoch(int y, int m, int d)
{
	y -= m <= 2;
	int era = y / 400;
	int yoe = y - era * 400;                                   // [0, 399]
	int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;  // [0, 365]
	int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;           // [0, 146096]
	return era * 146097 + doe - 719468;
}
/* Converts a Unix timestamp (number of seconds since the beginning of 1970
 * CE) to a Gregorian civil date-time tuple in GMT (UTC) time zone.
 *
 * This conforms to C89 (and C99...) and POSIX.
 *
 * This implementation works, and doesn't overflow for any sizeof(time_t).
 * It doesn't check for overflow/underflow in tm->tm_year output. Other than
 * that, it never overflows or underflows. It assumes that that time_t is
 * signed.
 *
 * This implements the inverse of the POSIX formula
 * (http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_15)
 * for all time_t values, no matter the size, as long as tm->tm_year doesn't
 * overflow or underflow. The formula is: tm_sec + tm_min*60 + tm_hour*3600
 * + tm_yday*86400 + (tm_year-70)*31536000 + ((tm_year-69)/4)*86400 -
 * ((tm_year-1)/100)*86400 + ((tm_year+299)/400)*86400.
 */
struct tm* gmtime_r(const time_t* timep, struct tm* tm) {
	const time_t ts = *timep;
	time_t t = ts / 86400;
	unsigned hms = ts % 86400;  /* -86399 <= hms <= 86399. This needs sizeof(int) >= 4. */
	time_t c, f;
	unsigned yday;  /* 0 <= yday <= 426. Also fits to an `unsigned short', but `int' is faster. */
	unsigned a;  /* 0 <= a <= 2133. Also fits to an `unsigned short', but `int' is faster. */
	if ((int)hms < 0) { --t; hms += 86400; }  /* Fix quotient and negative remainder if ts was negative (i.e. before year 1970 CE). */
	/* Now: -24856 <= t <= 24855. */
	tm->tm_sec = hms % 60;
	hms /= 60;
	tm->tm_min = hms % 60;
	tm->tm_hour = hms / 60;
	if (sizeof(time_t) > 4) {  /* Optimization. For int32_t, this would keep t intact, so we won't have to do it. This produces unreachable code. */
		f = (t + 4) % 7;
		if (f < 0) f += 7;  /* Fix negative remainder if (t + 4) was negative. */
		/* Now 0 <= f <= 6. */
		tm->tm_wday = f;
		c = (t << 2) + 102032;
		f = c / 146097;
		if (c % 146097 < 0) --f;  /* Fix negative remainder if c was negative. */
		--f;
		t += f;
		f >>= 2;
		t -= f;
		f = (t << 2) + 102035;
		c = f / 1461;
		if (f % 1461 < 0) --c;  /* Fix negative remainder if f was negative. */
	}
	else {
		tm->tm_wday = (t + 24861) % 7;  /* t + 24861 >= 0. */
		/* Now: -24856 <= t <= 24855. */
		c = ((t << 2) + 102035) / 1461;
	}
	yday = t - 365 * c - (c >> 2) + 25568;
	/* Now: 0 <= yday <= 425. */
	a = yday * 5 + 8;
	/* Now: 8 <= a <= 2133. */
	tm->tm_mon = a / 153;
	a %= 153;  /* No need to fix if a < 0, because a cannot be negative here. */
	/* Now: 2 <= tm->tm_mon <= 13. */
	/* Now: 0 <= a <= 152. */
	tm->tm_mday = 1 + a / 5;  /* No need to fix if a < 0, because a cannot be negative here. */
	/* Now: 1 <= tm->tm_mday <= 31. */
	if (tm->tm_mon >= 12) {
		tm->tm_mon -= 12;
		/* Now: 0 <= tm->tm_mon <= 1. */
		++c;
		yday -= 366;
	}
	else {  /* Check for leap year (in c). */
   /* Now: 2 <= tm->tm_mon <= 11. */
   /* 1903: not leap; 1904: leap, 1900: not leap; 2000: leap */
   /* With sizeof(time_t) == 4, we have 1901 <= year <= 2038; of these
	* years only 2000 is divisble by 100, and that's a leap year, no we
	* optimize the check to `(c & 3) == 0' only.
	*/
		if (!((c & 3) == 0 && (sizeof(time_t) <= 4 || c % 100 != 0 || (c + 300) % 400 == 0))) --yday;  /* These `== 0' comparisons work even if c < 0. */
	}
	tm->tm_year = c;  /* This assignment may overflow or underflow, we don't check it. Example: time_t is a huge int64_t, tm->tm_year is int32_t. */
	/* Now: 0 <= tm->tm_mon <= 11. */
	/* Now: 0 <= yday <= 365. */
	tm->tm_yday = yday;
	tm->tm_isdst = 0;
	return tm;
}


// It  does not modify broken-down time
long long timegm(struct tm const* t)
{
	int year = t->tm_year + 1900;
	int month = t->tm_mon;          // 0-11
	if (month > 11)
	{
		year += month / 12;
		month %= 12;
	}
	else if (month < 0)
	{
		int years_diff = (11 - month) / 12;
		year -= years_diff;
		month += 12 * years_diff;
	}
	int days_since_epoch = days_from_epoch(year, month + 1, t->tm_mday);

	return 60LL * (60LL * (24LL * days_since_epoch + (long long)t->tm_hour) + (long long)t->tm_min) + (long long)t->tm_sec;
}
/*
UTCClock::time_point UTCClock::fromDate(
    int year, int month, int day, int hour, int min, int sec, int usec)
{
  std::tm tm     = {0};
  tm.tm_year     = year - 1900;
  tm.tm_mon      = month - 1;
  tm.tm_mday     = day;
  tm.tm_hour     = hour;
  tm.tm_min      = min;
  tm.tm_sec      = sec;
  tm.tm_isdst    = -1;
  std::time_t tt = timegm(&tm);
  return from_time_t(tt) + chrono::microseconds(usec);
}


void UTCClock::toDate(const UTCClock::time_point &tp,
                      int &year,
                      int &month,
                      int &day,
                      int &hour,
                      int &min,
                      int &sec,
                      int &usec)
{
  std::time_t tt = to_time_t(tp);
  std::tm tm;
  gmtime_r(&tt, &tm);
  year  = tm.tm_year + 1900;
  month = tm.tm_mon + 1;
  day   = tm.tm_mday;
  hour  = tm.tm_hour;
  min   = tm.tm_min;
  chrono::microseconds leftover =
      tp - from_time_t(tt) + chrono::seconds(tm.tm_sec);
  sec = duration_cast<chrono::seconds>(leftover).count();
  usec = (leftover-chrono::seconds(sec)).count();
}
*/


long long date_string_to_time(std::string date)
{
	struct tm tm = { 0 }; // Important, initialize all members
	//int n = 0;
	//int year, mon, day, hour, min, sec;
	std::vector<std::string>  datetime, ddd, ttt;
	datetime = split(date, 'T');
	if (datetime.size() < 2)
	{
		datetime.clear();
		datetime = split(date, ' ');
	}

	ddd = split(datetime[0], '-');
	if (ddd.size() < 3)
	{
		ddd = split(datetime[0], '/');
	}
	// index for the year and mday
	//by default we assume date is written as yyyy-mm-ddTHH:MM:SS
	int indexyear = 0;
	int indexmday = 2;

	if (ddd[0].length() < 4 && ddd[2].length() == 4)//i.e. date is written as dd-mm-yyyy
	{
		int indexyear = 2;
		int indexmday = 0;
	}
	// If you write date like an American (mm-dd-yyyy). it wont work and I can't help

	tm.tm_year = std::stoi(ddd[indexyear]);

	tm.tm_mon = std::stoi(ddd[1]);

	tm.tm_mday = std::stoi(ddd[indexmday]);

	if (datetime.size() > 1)
	{

		ttt = split(datetime[1], ':');

		tm.tm_hour = std::stoi(ttt[0]);
		tm.tm_min = std::stoi(ttt[1]);
		if (ttt.size() == 3)
		{
			tm.tm_sec = std::stoi(ttt[2]);
		}
		else
		{
			tm.tm_sec = 0;
		}
	}
	else
	{
		tm.tm_hour = 0;
		tm.tm_min = 0;
		tm.tm_sec = 0;
	}




	//sscanf(date, "%d-%d-%dT%d:%d:%d %n", &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
	//	&tm.tm_hour, &tm.tm_min, &tm.tm_sec, &n);
	// If scan did not completely succeed or extra junk
	//if (n == 0 || date[n]) {
	//	return (time_t)-1;
	//}
	//tm.tm_isdst = 0; // Eforce output to be standard time. 
	tm.tm_mon--;      // Months since January
	// Assume 2 digit year if in the range 2000-2099, else assume year as given
	if (tm.tm_year >= 0 && tm.tm_year < 100) {
		tm.tm_year += 2000;
	}
	tm.tm_year -= 1900; // Years since 1900
	//UTCTime t1 = UTCClock::fromDate(year, mon, day, hour, min, sec, 0);
	long long t1 = timegm(&tm);
	return t1;
}

double date_string_to_s(std::string datetime, std::string refdate)
{
	//testime1(1);
	//testime2(2);

	//UTCTime ttime = date_string_to_time(datetime);
	//UTCTime reftime = date_string_to_time(refdate);

	long long ttime = date_string_to_time(datetime);
	long long reftime = date_string_to_time(refdate);

	//double diff = difftime(ttime, reftime);

	//std::chrono::microseconds timeDiff = ttime - reftime;

	//double diff = ((double) duration_cast<std::chrono::milliseconds>(ttime - reftime).count())/1000.0;

	double diff = (double)(ttime - reftime);

	return diff;
}

// Read time string. If it is a valid datetime string return s from reftime otherwise return a foat of seconds 
double readinputtimetxt(std::string input, std::string & refdate)
{
	std::string date = trim(input, " ");
	double timeinsec;
	//check if string contains a T a marker of 
	std::vector<std::string>  datetime = split(date, 'T');

	if (datetime.size() > 1)
	{
		//likely a datetime
		if (refdate.empty())
		{
			refdate = date;
		}
		timeinsec = date_string_to_s(date, refdate);


	}
	else
	{
		//Likely a float
		timeinsec = std::stod(datetime[0]);
	}

	return timeinsec;
}


bool testime1(int hour)
{
	bool test = false;
	double eps = 1e-7;

	for (int iy = 1400; iy <= 2800; iy++)
	{
		tm tm1, tm2;

		tm1.tm_year = iy;
		tm2.tm_year = iy;

		tm1.tm_mday = 1;
		tm2.tm_mday = 1;

		tm1.tm_mon = 0;
		tm2.tm_mon = 0;

		tm1.tm_hour = 0;
		tm2.tm_hour = hour;

		tm1.tm_min = 0;
		tm2.tm_min = 0;

		tm1.tm_sec = 0;
		tm2.tm_sec = 0;

		//UTCTime t1 = UTCClock::fromDate(iy, 1, 1, 0, 0, 0, 0);
		//UTCTime t2 = UTCClock::fromDate(iy, 1, 1, hour, 0, 0, 0);

		long long t1 = timegm(&tm1);
		long long t2 = timegm(&tm2);




		//double dt12 = ((double)duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0;
		double dt12 = (double)(t2 - t1);
		test = abs(dt12 - (hour * 3600.0)) < eps;
		if (!test)
		{
			printf("Failed datetime calculation: year=%d\n", iy);
			break;
		}
	}


	return test;
}
bool testime2(int hour)
{
	bool test = false;
	

	for (int iy = 1970; iy <= 2400; iy++)
	{
		//UTCTime t1 = UTCClock::fromDate(1970, 1, 1, 0, 0, 0, 0);
		//UTCTime t2 = UTCClock::fromDate(iy, 1, 1, hour, 0, 0, 0);
		tm tm1, tm2;

		tm1.tm_year = 1970;
		tm2.tm_year = iy;

		tm1.tm_mday = 1;
		tm2.tm_mday = 1;

		tm1.tm_mon = 0;
		tm2.tm_mon = 0;

		tm1.tm_hour = 0;
		tm2.tm_hour = hour;

		tm1.tm_min = 0;
		tm2.tm_min = 0;

		tm1.tm_sec = 0;
		tm2.tm_sec = 0;

		//UTCTime t1 = UTCClock::fromDate(iy, 1, 1, 0, 0, 0, 0);
		//UTCTime t2 = UTCClock::fromDate(iy, 1, 1, hour, 0, 0, 0);

		long long t1 = timegm(&tm1);
		long long t2 = timegm(&tm2);



		//double dt12 = ((double)duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0;

		test = t2>t1;
		if (!test)
		{

			int dse = days_from_epoch(tm2.tm_year, tm2.tm_mon + 1, tm2.tm_mday);
			printf("Failed datetime calculation greater than: year=%d\n", iy);
			printf("dse=%d\nt1 = %lld;\n t2=%lld\n",dse, t1, t2);
			break;
		}
	}


	return test;
}
