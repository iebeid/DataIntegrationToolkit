import sys
import traceback
from typing import Optional
from toolkit.context import IngestionContext


def print_table_records(records):
    if not records:
        print("(0 rows returned)")
        return
    if isinstance(records, list) and isinstance(records[0], dict):
        headers = list(records[0].keys())
        # Truncate values for display
        rows = [[str(r.get(h, ""))[:40] for h in headers] for r in records]
        col_widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        print(header_line)
        print(sep_line)
        for row in rows:
            print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))
        print(f"({len(records)} rows)")
    else:
        print(records)


def run_cli(lakehouse_root: str = "./lakehouse"):
    print("================================================================================")
    print("  DATA INTEGRATION TOOLKIT (DITK) - LAKEHOUSE INGESTION SQL SHELL")
    print("  Type SQL statements ending with ';'. Type 'exit;' or 'quit;' to exit.")
    print("================================================================================")

    ctx = IngestionContext(lakehouse_root=lakehouse_root)
    buffer = ""

    while True:
        try:
            prompt = "lakehouse> " if not buffer else "      ...> "
            line = input(prompt)
            if not line.strip() and not buffer:
                continue

            buffer += " " + line.strip()

            if buffer.strip().endswith(";"):
                stmt = buffer.strip()
                buffer = ""

                cmd = stmt.rstrip(";").strip()
                if cmd.lower() in ["exit", "quit"]:
                    print("Exiting Lakehouse Shell. Goodbye!")
                    break
                if cmd.lower() == "help":
                    print("""
Supported Commands:
  SHOW TABLES [bronze|silver|gold];
  DESCRIBE TABLE <name>;
  INGEST FROM '<path>' INTO bronze.<name> WITH (format = 'csv', delimiter = ',');
  CREATE STAGE silver.<name> FROM bronze.<name> WITH (key = 'RecID', stopword_threshold = 0.05);
  CREATE BLOCKS silver.<pairs> FROM silver.<staged> WITH (key = 'RecID', max_block_size = 200);
  RESOLVE ENTITIES INTO gold.<resolved> FROM silver.<staged> USING BLOCKS silver.<pairs> WITH (threshold = 0.5);
  PROJECT GRAPH gold.<graph> FROM <source> WITH (sequence_column = 'seq', n = 3);
  SELECT <cols> FROM <name> [LIMIT <n>];
                    """)
                    continue

                res = ctx.sql(stmt)
                if hasattr(res, "print_report"):
                    res.print_report()
                elif hasattr(res, "summary"):
                    print_table_records([res.summary()])
                elif isinstance(res, list):
                    print_table_records(res)
                elif isinstance(res, dict):
                    print_table_records([res])
                else:
                    print(res)

        except KeyboardInterrupt:
            print("\nQuery interrupted.")
            buffer = ""
        except EOFError:
            break
        except Exception as e:
            print(f"ERROR: {e}")
            buffer = ""


def handle_ai_cli(args):
    command = args[1].lower()
    if len(args) < 3:
        print(f"Usage: python -m toolkit.cli {command} <dataset_path> [--execute] [--api-key <key>]")
        return
    dataset_path = args[2]
    auto_exec = "--execute" in args
    api_key = None
    if "--api-key" in args:
        idx = args.index("--api-key")
        if idx + 1 < len(args):
            api_key = args[idx + 1]

    ctx = IngestionContext(lakehouse_root="./lakehouse")
    if command == "ai-profile":
        plan = ctx.ai_profile(dataset_path, api_key=api_key)
    else:
        plan = ctx.ai_route(dataset_path, auto_execute=auto_exec, api_key=api_key)

    print("\n" + "=" * 80)
    print("  AI / GEMINI DATASET PROFILING & INGESTION ROUTING REPORT")
    print("=" * 80)
    for k, v in plan.to_dict().items():
        print(f"  * {k:<22}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("ai-profile", "ai-route"):
        handle_ai_cli(sys.argv)
    else:
        root = sys.argv[1] if len(sys.argv) > 1 else "./lakehouse"
        run_cli(lakehouse_root=root)
