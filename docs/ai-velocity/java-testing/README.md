# Java Unit Testing Skill

This skill enables AI agents to generate high-quality unit and integration test cases for Java services using JUnit 5, Mockito, and Spring Boot Test.

## Quick Start

### When to Use This Skill

- User requests: "Write test cases for this class", "Add unit tests", "Test coverage"
- After implementing new features or bug fixes
- When reviewing code that lacks test coverage
- When refactoring code that needs test validation

### How to Use

1. **Point to the class/service** you want to test
2. **Request test generation**: "Generate unit tests for this class"
3. **The AI will automatically**:
   - Analyze the code structure
   - Identify dependencies and behaviors
   - Generate comprehensive test cases
   - Follow existing test patterns in your codebase

## Prerequisites

**Java 17+ is required.** Verify installation:

```bash
java -version  # Should show Java 17 or later
```

**If Java is not installed or JAVA_HOME is not set:**

### macOS
```bash
# Install Java
brew install openjdk@17

# Set JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
```

### Linux
```bash
# Install Java
sudo apt install openjdk-17-jdk  # Ubuntu/Debian
# or
sudo yum install java-17-openjdk-devel  # CentOS/RHEL

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
```

### Windows
1. Download Java 17 from https://adoptium.net/
2. Set environment variable: `JAVA_HOME=C:\Program Files\Java\jdk-17`
3. Add to PATH: `%JAVA_HOME%\bin`

**See [SKILL.md](SKILL.md#java-runtime-setup) for detailed setup instructions.**

## Running Tests

### Using Gradle

```bash
# Verify Java is configured
./gradlew --version

# Run all tests
./gradlew test

# Run specific test class
./gradlew test --tests "tech.vance.goblin.service.ServiceClassTest"

# Run with coverage
./gradlew test jacocoTestReport

# View coverage report
open build/reports/jacoco/test/html/index.html
```

### Using IDE

- **IntelliJ IDEA**: Right-click test class → Run 'TestClassName'
- **VS Code**: Use Java Test Runner extension
- **Eclipse**: Right-click → Run As → JUnit Test

## Test Structure

Tests are organized in `src/test/java/` mirroring the main source structure:

```
src/
├── main/java/tech/vance/goblin/service/OrderService.java
└── test/java/tech/vance/goblin/service/OrderServiceTest.java
```

## Key Features

- ✅ **SOLID Principles**: Single responsibility, proper dependency injection
- ✅ **DRY Principle**: Reusable test fixtures and helper methods
- ✅ **AAA Pattern**: Arrange-Act-Assert structure
- ✅ **Comprehensive Coverage**: Happy path, edge cases, error scenarios
- ✅ **Spring Boot Support**: Unit and integration test patterns
- ✅ **Well Documented**: Clear test names and JavaDoc comments

## Examples

See the skill documentation (`SKILL.md`) for detailed examples and patterns.
