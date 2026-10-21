# Data

Data files are not committed. Put them here locally:

| Folder | Contents |
| --- | --- |
| `raw/` | Downloaded sources as received: SYNOP archives, OGIMET text pages, soundings (IGRA2 / Wyoming), ERA5 / GFS fields |
| `processed/` | Decoded, cleaned tables used by the code, e.g. `synop.parquet` |

## Expected SYNOP table

One row per report, with at least:

| Column | Meaning |
| --- | --- |
| `WMO_ID` | 5-digit station index |
| `ANO`, `MES`, `DIA`, `HORA` | year, month, day, hour (UTC) |
| `visibility_m` | horizontal visibility (m) |

Other decoded columns (temperature, dewpoint, wind, present weather, cloud base) are kept as they are.

## Station metadata

Coordinates and elevations for 36 Czech SYNOP stations ship with the package
(`src/fogfusion/resources/stations_cz.csv`), parsed from an OGIMET query page.
For other regions, save an OGIMET SYNOP page as text and parse it:

```python
from fogfusion.io import parse_ogimet_file
stations = parse_ogimet_file("data/raw/ogimet_page.txt")
```

OGIMET does not allow automated scraping, so save the page from a browser.
