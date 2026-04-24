# Feature: Type Field (Phase 1)

Add a `type` field (`uint8_t`) to particle data, enabling identification of particle species.

## Scenarios

### Scenario: Type field exists in field registry
Given: a ParticlesData instance
When: the field names are queried
Then: "type" appears in `fieldNames`
And: its FieldVariant resolves to `FieldVector<uint8_t>*`
Test: `TEST(TypeField, existsInFieldRegistry)`

### Scenario: Type field included in dataTuple
Given: a ParticlesData instance with allocated type field
When: `dataTuple()` is called
Then: the tuple contains a reference to the type FieldVector
And: `data()` returns a FieldVariant array that includes the type field
Test: `TEST(TypeField, includedInDataTuple)`

### Scenario: Type field resized with other fields
Given: a ParticlesData instance with type set as conserved
When: `resize(N)` is called
Then: `type.size() == N`
And: `type.size() == x.size()` (uniform array size — INV-13)
Test: `TEST(TypeField, resizedWithOtherFields)`

### Scenario: Type field preserves values on resize
Given: a ParticlesData with 100 particles, type[i] set to i % 2
When: `resize(200)` is called
Then: type[0..99] retain their original values
Test: `TEST(TypeField, preservesValuesOnResize)`

### Scenario: Default type is gas for backward compatibility
Given: an existing initializer (e.g., Sedov) that does not set type
When: particles are initialized
Then: all particles have type = 1 (gas)
Test: `TEST(TypeField, defaultTypeIsGas)`

### Scenario: Type field activatable as conserved
Given: a ParticlesData instance
When: `setConserved("type")` is called
Then: the type field is marked as conserved (allocated, never released)
Test: `TEST(TypeField, activatableAsConserved)`

### Scenario: Type field accessible by name
Given: a ParticlesData with type field allocated
When: `getFieldIndex("type", fieldNames)` is called
Then: the returned index corresponds to the type FieldVector in `data()`
Test: `TEST(TypeField, accessibleByName)`

### Scenario: Type field writable to HDF5
Given: a ParticlesData with mixed types (some 0, some 1)
When: output fields include "type" and data is written to file
Then: the file contains a "type" dataset with correct uint8_t values
Test: `TEST(TypeFieldIO, writableToHdf5)`

### Scenario: Type field readable from HDF5
Given: an HDF5 file with a "type" dataset
When: the file is loaded into ParticlesData
Then: the type field values match the file contents
Test: `TEST(TypeFieldIO, readableFromHdf5)`

## Invariants tested
- INV-13 (uniform array sizes): type array same size as all others
- INV-2 (type immutable): type values preserved across resize
