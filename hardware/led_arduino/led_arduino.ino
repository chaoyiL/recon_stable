#include <Adafruit_NeoPixel.h>

const uint8_t LED_PIN = 6;
// 包含开头空出的灯珠；限制两份 RGB 缓冲区的内存占用，适配 Uno。
const uint16_t MAX_LED_COUNT = 128;
const uint8_t HEADER0 = 0xAA;
const uint8_t HEADER1 = 0x55;
const unsigned long BYTE_TIMEOUT_MS = 200;

// AA 55 + uint16 小端总数量 N + N*3 字节 RGB + XOR(数量字节及 RGB)。
Adafruit_NeoPixel strip(0, LED_PIN, NEO_GRB + NEO_KHZ800);
uint8_t frame[MAX_LED_COUNT * 3];

enum ParserState { HEADER_0, HEADER_1, COUNT_LOW, COUNT_HIGH, DATA, CHECKSUM };
ParserState state = HEADER_0;
uint16_t ledCount = 0;
uint16_t frameBytes = 0;
uint16_t pos = 0;
uint8_t checksum = 0;
unsigned long lastByteTime = 0;

void resetParser() {
  state = HEADER_0;
  ledCount = 0;
  frameBytes = 0;
  pos = 0;
  checksum = 0;
}

void applyFrame() {
  if (strip.numPixels() != ledCount) {
    // 先按旧长度熄灭，防止缩短灯带后尾部仍保留上一次的颜色。
    strip.clear();
    strip.show();
    strip.updateLength(ledCount);
    if (strip.numPixels() != ledCount) {
      Serial.write('M');
      return;
    }
  }
  for (uint16_t i = 0; i < ledCount; ++i) {
    strip.setPixelColor(i, strip.Color(frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]));
  }
  strip.show();
  Serial.write('K');
}

void setup() {
  strip.begin();
  // 复位后也清掉上次运行留下的颜色，再由首个有效帧设置实际长度。
  strip.updateLength(MAX_LED_COUNT);
  strip.show();
  Serial.begin(115200);
}

void loop() {
  if (state != HEADER_0 && millis() - lastByteTime > BYTE_TIMEOUT_MS) {
    resetParser();
  }
  while (Serial.available() > 0) {
    uint8_t value = (uint8_t)Serial.read();
    lastByteTime = millis();
    switch (state) {
      case HEADER_0:
        if (value == HEADER0) state = HEADER_1;
        break;
      case HEADER_1:
        if (value == HEADER1) {
          state = COUNT_LOW;
          checksum = 0;
        } else {
          state = (value == HEADER0) ? HEADER_1 : HEADER_0;
        }
        break;
      case COUNT_LOW:
        ledCount = value;
        checksum = value;
        state = COUNT_HIGH;
        break;
      case COUNT_HIGH:
        ledCount |= (uint16_t)value << 8;
        checksum ^= value;
        if (ledCount == 0 || ledCount > MAX_LED_COUNT) {
          Serial.write('L');
          resetParser();
        } else {
          frameBytes = ledCount * 3;
          pos = 0;
          state = DATA;
        }
        break;
      case DATA:
        frame[pos++] = value;
        checksum ^= value;
        if (pos == frameBytes) state = CHECKSUM;
        break;
      case CHECKSUM:
        if (value == checksum) {
          applyFrame();
        } else {
          Serial.write('E');
        }
        resetParser();
        break;
    }
  }
}
