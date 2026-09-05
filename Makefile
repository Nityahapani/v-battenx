BUILD_DIR ?= build

debug:
	mkdir -p $(BUILD_DIR)/debug
	cd $(BUILD_DIR)/debug && cmake ../.. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

release:
	mkdir -p $(BUILD_DIR)/release
	cd $(BUILD_DIR)/release && cmake ../.. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

asan:
	mkdir -p $(BUILD_DIR)/asan
	cd $(BUILD_DIR)/asan && cmake ../.. -DCMAKE_BUILD_TYPE=Debug \
	    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" && make -j$(nproc)

test: release
	cd $(BUILD_DIR)/release && ctest --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

.PHONY: debug release asan test clean
