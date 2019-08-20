#pragma once

#include "llvm/Pass.h"

namespace llvm {
Pass *createMarkAllDeviceForInlinePass();
Pass *createLinkDeviceSupportPass();
Pass *createInstrumentDevicePass(bool, bool);

Pass *createLinkHostSupportPass();
Pass *createInstrumentHostPass();
}
