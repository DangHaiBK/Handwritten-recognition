################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../X-CUBE-AI/App/aiSystemPerformance.c \
../X-CUBE-AI/App/aiTestHelper.c \
../X-CUBE-AI/App/aiTestUtility.c \
../X-CUBE-AI/App/app_x-cube-ai.c \
../X-CUBE-AI/App/lc_print.c \
../X-CUBE-AI/App/mlp.c \
../X-CUBE-AI/App/mlp_data.c \
../X-CUBE-AI/App/mlp_data_params.c \
../X-CUBE-AI/App/syscalls.c 

OBJS += \
./X-CUBE-AI/App/aiSystemPerformance.o \
./X-CUBE-AI/App/aiTestHelper.o \
./X-CUBE-AI/App/aiTestUtility.o \
./X-CUBE-AI/App/app_x-cube-ai.o \
./X-CUBE-AI/App/lc_print.o \
./X-CUBE-AI/App/mlp.o \
./X-CUBE-AI/App/mlp_data.o \
./X-CUBE-AI/App/mlp_data_params.o \
./X-CUBE-AI/App/syscalls.o 

C_DEPS += \
./X-CUBE-AI/App/aiSystemPerformance.d \
./X-CUBE-AI/App/aiTestHelper.d \
./X-CUBE-AI/App/aiTestUtility.d \
./X-CUBE-AI/App/app_x-cube-ai.d \
./X-CUBE-AI/App/lc_print.d \
./X-CUBE-AI/App/mlp.d \
./X-CUBE-AI/App/mlp_data.d \
./X-CUBE-AI/App/mlp_data_params.d \
./X-CUBE-AI/App/syscalls.d 


# Each subdirectory must supply rules for building sources it contributes
X-CUBE-AI/App/%.o X-CUBE-AI/App/%.su: ../X-CUBE-AI/App/%.c X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_HOST/App -I../USB_HOST/Target -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Middlewares/ST/STM32_USB_Host_Library/Core/Inc -I../Middlewares/ST/STM32_USB_Host_Library/Class/CDC/Inc -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -I../X-CUBE-AI/Target -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-X-2d-CUBE-2d-AI-2f-App

clean-X-2d-CUBE-2d-AI-2f-App:
	-$(RM) ./X-CUBE-AI/App/aiSystemPerformance.d ./X-CUBE-AI/App/aiSystemPerformance.o ./X-CUBE-AI/App/aiSystemPerformance.su ./X-CUBE-AI/App/aiTestHelper.d ./X-CUBE-AI/App/aiTestHelper.o ./X-CUBE-AI/App/aiTestHelper.su ./X-CUBE-AI/App/aiTestUtility.d ./X-CUBE-AI/App/aiTestUtility.o ./X-CUBE-AI/App/aiTestUtility.su ./X-CUBE-AI/App/app_x-cube-ai.d ./X-CUBE-AI/App/app_x-cube-ai.o ./X-CUBE-AI/App/app_x-cube-ai.su ./X-CUBE-AI/App/lc_print.d ./X-CUBE-AI/App/lc_print.o ./X-CUBE-AI/App/lc_print.su ./X-CUBE-AI/App/mlp.d ./X-CUBE-AI/App/mlp.o ./X-CUBE-AI/App/mlp.su ./X-CUBE-AI/App/mlp_data.d ./X-CUBE-AI/App/mlp_data.o ./X-CUBE-AI/App/mlp_data.su ./X-CUBE-AI/App/mlp_data_params.d ./X-CUBE-AI/App/mlp_data_params.o ./X-CUBE-AI/App/mlp_data_params.su ./X-CUBE-AI/App/syscalls.d ./X-CUBE-AI/App/syscalls.o ./X-CUBE-AI/App/syscalls.su

.PHONY: clean-X-2d-CUBE-2d-AI-2f-App

