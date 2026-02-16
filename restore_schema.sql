-- Restore Images table
create table if not exists public.Images (
  id uuid default gen_random_uuid() primary key,
  original_name text,
  file_path text not null, -- Path in Supabase Storage
  public_url text, -- Publicly accessible URL if applicable
  user_id uuid default auth.uid(), -- Owner
  file_type text, -- Mime type
  file_size bigint,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable RLS for Images
alter table public.Images enable row level security;

-- Policy: Allow public read access to images (adjust if private)
create policy "Public Read Access"
on public.Images for select
to public
using ( true );

-- Policy: Allow authenticated users to upload (insert)
create policy "Authenticated Insert"
on public.Images for insert
to authenticated
with check ( true ); -- You might want to restrict to match auth.uid()

-- Policy: Users can update their own images
create policy "Owner Update"
on public.Images for update
to authenticated
using ( auth.uid() = user_id );

-- Policy: Users can delete their own images
create policy "Owner Delete"
on public.Images for delete
to authenticated
using ( auth.uid() = user_id );


-- Restore user_resource_usage table
create table if not exists public.user_resource_usage (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id), -- Optional: strict FK
  resource_type text not null, -- e.g., 'llm_tokens', 'search_api'
  amount numeric default 0, -- Use numeric for flexibility (tokens, dollars)
  details jsonb, -- Store extra info like provider, model
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable RLS for user_resource_usage
alter table public.user_resource_usage enable row level security;

-- Policy: Users can view their own usage
create policy "User View Own Usage"
on public.user_resource_usage for select
to authenticated
using ( auth.uid() = user_id );

-- Policy: Service role or admin can insert (no public insert usually)
-- Note: Service role bypasses RLS, so explicit policy might not be needed for backend scripts,
-- but good to have if inserting via client with specialized claims.
-- For now, we assume usage is tracked via backend.
