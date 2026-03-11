create table if not exists public.detections (
  id bigserial primary key,
  timestamp timestamptz not null default now(),
  category text not null,
  detected integer not null default 0,
  defect integer not null default 0,
  good integer not null default 0,
  mode text,
  expected_count integer
);

alter table public.detections enable row level security;

create policy "public_read_detections"
on public.detections
for select
to anon
using (true);
