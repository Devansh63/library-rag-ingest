
## Step 1 - Create an Account

1. Go to: https://isbndb.com
2. Click **Sign Up** in the top right
3. Fill in name, email, and password
4. Verify your email address

---

## Step 2 - Subscribe to the Basic Plan

The free tier does not include API access. You need at least the Basic plan.

1. After logging in, go to: https://isbndb.com/isbn-database-api-access
2. Click **Subscribe** on the **Basic** plan
3. just get is for one week for the free trial MAKE SURE TO CANCEL IS AFTER A WEEK.

**Basic plan specs:**
- $10/month (as of April 2026)
- 5,000 API calls per day
- Each ISBN = 1 call (even in bulk requests)
- 100 ISBNs per bulk request
- Result: 50 bulk API calls/day = 5,000 books/day

**Note:** The "call" quota counts ISBNs, not HTTP requests. If you send a batch of 100 ISBNs, that costs 100 calls even though it is 1 HTTP request. This is why the enrichment script pre-checks quota before each batch and stops when remaining - buffer < batch_size.

---

## Step 3 - Get Your API Key

1. Log in at https://isbndb.com
2. Go to your profile or dashboard (top right, click your username)
3. Find the **API Key** section
4. Copy the key - it looks like: `68546_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`
