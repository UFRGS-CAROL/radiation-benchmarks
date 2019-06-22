/**
 * @file jtx1inst.h
 * @author cs
 * @brief This header file collects all the header files specific to Jetson TX1.
 */

#ifndef JTX1INST_H_
#define JTX1INST_H_

//#include <stdio.h>
//#include <unistd.h>
//#include <string.h>
//#include <fcntl.h>
//#include <errno.h>
#include <sys/ioctl.h>

/**
 * @file jtx1pow.h
 * @author cs
 * @brief This header file contains a declaration of the function 
 * for accessing on-board and on-module INA3221's values together 
 * with enumerated types.
 */

#define MAX_BUFF 128
//#define SYSFS_INA3321_PATH "/sys/class/i2c-dev/i2c-1/device"
#define SYSFS_INA3321_PATH "/sys/devices/3160000.i2c/i2c-0"
/**
 * @brief Enumeration indexing each INA3221's input.
 */
typedef enum jtx2_rails {
	VDD_SYS_GPU = 0,
	VDD_SYS_SOC,
	VDD_4V0_WIFI,
	VDD_IN,
	VDD_SYS_CPU,
	VDD_SYS_DDR,
	VDD_MUX,
	VDD_5V0_IO_SYS,
	VDD_3V3_SYS,
	VDD_3V3_IO_SLP,
	VDD_1V8_IO,
	VDD_3V3_SYS_M2

} jtx2_rail;

/*
 typedef enum jtx1_rails {
 VDD_IN = 0, ///< main module power input
 VDD_GPU, ///< GPU Power rail
 VDD_CPU, ///< CPU Power rail
 VDD_MUX, ///< main carrier board power input
 VDD_5V_IO_SYS, ///< main carrier board 5V supply
 VDD_3V3_SYS, ///< main carrier board 3.3V supply
 VDD_3V3_IO, ///< carrier board 3.3V Sleep supply
 VDD_1V8_IO, ///< main carrier board 1.8V supply
 VDD_M2_IN ///< 3.3V supply for M.2 Key E connector
 } jtx1_rail;
 */
/**
 * @brief Enumeration indexing each type of measurement.
 */
/*
 typedef enum jtx1_rail_types {
 VOLTAGE = 0, ///< Voltage given in milli-volts or mV
 POWER, ///< Power given in milli-watts or mW
 CURRENT ///< Power given in milli-amps mA
 } jtx1_rail_type;
 */
typedef enum jtx2_rail_types {
	VOLTAGE = 0, ///< Voltage given in milli-volts or mV
	POWER, ///< Power given in milli-watts or mW
	CURRENT ///< Power given in milli-amps mA
} jtx2_rail_type;

/**
 * @brief Read on-board and on-module INA3221's values
 * 
 * Use sysf files to access on-board INA3221 sensor 
 * and userspace I2C to access on-module INA3221 sensor
 * and read power, current, and voltage information.
 *
 * @param rail Indexed by ::jtx1_rail
 * @param type Either VOLTAGE, POWER or CURRENT. See ::jtx1pow_ina3321_measure
 * @param *val Output's reference
 */
void jtx1_get_ina3221(jtx2_rail rail, jtx2_rail_type type, unsigned int *val);

/**
 * @file jtx1rate.h
 * @author cs
 * @brief This header file contains the functions for setting and reading
 * Jetson TX1's GPU and EMC frequncies.
 */

/**
 * @brief Units with adjustable operating frequency
 */
typedef enum jtx1_units {
	EMC_RATE = 0, ///< external memory controller (EMC)
	GPU_RATE, ///< graphics processing unit (GPU)
	CPU0_RATE, ///< first core of central processing unit (CPU)
	CPU1_RATE, ///< second core of CPU
	CPU2_RATE, ///< third core of CPU
	CPU3_RATE, ///< fourth core of CPU
} jtx1_unit;

/**
 * @brief Read operating frequency
 * @param unit See ::jtx1_unit
 * @param *rate Output's reference
 */
void jtx1_get_rate(const jtx1_unit unit, unsigned long *rate);

/**
 * @brief Set operating frequency
 * @param unit See ::jtx1_unit
 * @param rate Operating frequency 
 */
void jtx1_set_rate(const jtx1_unit unit, const unsigned long rate);

/**
 * @file jtx1temp.h
 * @author cs
 * @brief This header file contains the functions for reading
 * Jetson TX1's values of thermal zones.
 */

#define MAX_BUFF 128
//#define SYSFS_TEMP_PATH "/sys/class/thermal"	// TX1
#define SYSFS_TEMP_PATH "/sys/devices/virtual/thermal" // TX2

/**
 * @brief Thermal zones index
 */
typedef enum jtx1_tzones {
	A0 = 0, ///< on-chip thermal zone (mC)
	CPU, ///< on-chip thermal zone (mC)
	GPU, ///< on-chip thermal zone (mC)
	PLL, ///< on-chip thermal zone (mC)
	PMIC, ///< on-chip thermal zone (mC)
	TDIODE, ///< on-module thermal zone (mC)
	TBOARD, ///< on-module thermal zone (mC)
	FAN ///< on-chip thermal zone (mC)
} jtx1_tzone;

/**
 * @brief Read on-chip and on-module temperatures.
 *
 * @param zone Indexed by ::jtx1_tzone
 * @param *temperature Output's reference
 */
void jtx1_get_temp(jtx1_tzone zone, unsigned int *temperature);

/**
 * @file jtx1par.h
 * @author cs
 * @brief This header file contains Jetson TX1 data and parameters.
 */

/**
 * @brief Available frequencies of Jetson TX1's GPU
 */
static const unsigned long jtx1_gpu_freqs[] = { 76800000, 153600000, 230400000,
		307200000, 384000000, 460800000, 537600000, 614400000, 691200000,
		768000000, 844800000, 921600000, 998400000 }; // [Hz]

/**
 * @brief Available frequencies of Jetson TX1's EMC
 */
static const unsigned long jtx1_emc_freqs[] = {
		/*40800000, 68000000, 102000000,*/ // problematic
		204000000, 408000000, 665600000, 800000000, 1065600000, 1331200000,
		1600000000 }; // [Hz]

#endif
